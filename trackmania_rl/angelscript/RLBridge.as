// RLBridge.as
// -----------------------------------------------------------------------------
// AngelScript plugin that exposes TrackMania Nations Forever simulation data to
// an external reinforcement learning agent through a TCP socket.  The Python
// side of the project listens on ``cfg.tmiface.server_name:cfg.tmiface.bridge_port``
// and expects newline separated JSON payloads.
//
// The script intentionally mirrors the high level information that used to be
// provided by the legacy Python TMInterface bindings so that the rest of the
// RL stack can operate unchanged.
// -----------------------------------------------------------------------------

PluginInfo@ GetPluginInfo()
{
    PluginInfo info;
    info.Name = "TMNF RL Bridge";
    info.Author = "TMNF-AI";
    info.Version = "1.0";
    info.Description = "Streams simulation state to an external RL agent.";
    return info;
}

namespace RLBridge
{
    Net::Socket@ g_Socket;
    string g_ReadBuffer;

    float g_LastSteer = 0.0f;
    float g_LastGas = 0.0f;
    bool g_LastBrake = false;

    bool g_PreventFinish = true;

    const string kHost = "127.0.0.1";
    const uint16 kPort = 54545;

    void Main()
    {
        startnew(CoroutineFunc(ConnectLoop));
    }

    void OnDisabled()
    {
        if (g_Socket !is null && g_Socket.IsConnected())
        {
            g_Socket.Close();
        }
        @g_Socket = null;
        g_ReadBuffer = "";
    }

    void OnRunStep(SimulationManager@ sim)
    {
        if (sim is null)
            return;

        EnsureConnection();
        if (g_Socket is null || !g_Socket.IsConnected())
            return;

        TM::SceneVehicleCar@ car = sim.SceneVehicleCar;
        if (car is null)
            return;

        TM::PlayerInfo@ player = sim.PlayerInfo;

        SendState(sim, car, player);
        PollIncoming(sim, car);
    }

    void ConnectLoop()
    {
        while (true)
        {
            EnsureConnection();
            sleep(500);
        }
    }

    void EnsureConnection()
    {
        if (g_Socket !is null && g_Socket.IsConnected())
            return;

        if (g_Socket is null)
            @g_Socket = Net::CreateTCPSocket();

        if (g_Socket.Connect(kHost, kPort))
        {
            g_Socket.Blocking = false;
            SendLog("Connected to RL agent");
        }
    }

    void SendState(SimulationManager@ sim, TM::SceneVehicleCar@ car, TM::PlayerInfo@ player)
    {
        Json::Value root = Json::Object();
        root["type"] = "state";

        // Position / orientation
        vec3 position = vec3();
        float yaw = 0.0f;

        TM::HmsDyna@ dyna = car.Dyna;
        if (dyna !is null)
        {
            TM::HmsStateDyna@ state = dyna.Forward;
            if (state !is null)
            {
                position = state.Location.Translation;
                yaw = Math::Atan2(state.Location.XX, state.Location.ZX);
            }
        }

        root["position"] = Json::Array();
        root["position"].Add(position.x);
        root["position"].Add(position.y);
        root["position"].Add(position.z);
        root["yaw"] = yaw;

        // Speed / velocity
        float speed = 0.0f;
        if (player !is null)
        {
            speed = player.DisplaySpeed;
            root["race_time"] = player.RaceTime;
            root["cp_index"] = player.CurCheckpoint;
            root["has_any_lateral_contact"] = car.HasAnyLateralContact;
            root["race_finished"] = player.RaceFinished;
        }
        else if (dyna !is null && dyna.Forward !is null)
        {
            vec3 vel = dyna.Forward.LinearVelocity;
            speed = vel.Length();
        }
        root["speed"] = speed;

        // Wheel contact data
        Json::Value wheelContacts = Json::Object();
        for (uint i = 0; i < car.Wheels.Length; i++)
        {
            auto@ wheel = car.Wheels[i];
            bool grounded = wheel !is null && wheel.RealTimeState.HasGroundContact;
            wheelContacts["wheel_" + i] = grounded;
        }
        root["wheel_ground_contact"] = wheelContacts;
        root["nb_wheels_grounded"] = car.NbWheelsGrounded;

        // Just respawned flag – true when simulation reports reset
        root["just_respawned"] = sim.PlayerJustRespawned;

        string payload = Json::Write(root) + "\n";
        array<uint8> data(payload.Length);
        for (uint i = 0; i < payload.Length; i++)
        {
            data[i] = payload[i];
        }
        g_Socket.Send(data, data.Length);
    }

    void PollIncoming(SimulationManager@ sim, TM::SceneVehicleCar@ car)
    {
        if (g_Socket.Available() <= 0)
            return;

        string chunk = g_Socket.Receive(g_Socket.Available());
        if (chunk.Length == 0)
            return;

        g_ReadBuffer += chunk;

        while (true)
        {
            int newline = g_ReadBuffer.IndexOf('\n');
            if (newline < 0)
                break;

            string line = g_ReadBuffer.SubStr(0, newline);
            g_ReadBuffer = g_ReadBuffer.SubStr(newline + 1);
            if (line.Length == 0)
                continue;

            Json::Value msg = Json::Parse(line);
            string msgType = msg.HasKey("type") ? string(msg["type"]) : "inputs";
            if (msgType == "inputs")
            {
                ApplyInputs(car, msg);
            }
            else if (msgType == "command")
            {
                HandleCommand(sim, msg);
            }
        }
    }

    void ApplyInputs(TM::SceneVehicleCar@ car, Json::Value&in msg)
    {
        g_LastSteer = float(msg.Get("steer", g_LastSteer));
        g_LastGas = float(msg.Get("gas", g_LastGas));
        g_LastBrake = bool(msg.Get("brake", g_LastBrake));

        car.InputSteer = Math::Clamp(g_LastSteer, -1.0f, 1.0f);
        car.InputGasPedal = Math::Clamp(g_LastGas, 0.0f, 1.0f);
        car.InputBrake = g_LastBrake;
    }

    void HandleCommand(SimulationManager@ sim, Json::Value&in msg)
    {
        string name = msg.Get("name", "");
        if (name == "set_game_speed")
        {
            float speed = float(msg.Get("value", 1.0f));
            sim.GameSpeed = speed;
        }
        else if (name == "respawn")
        {
            bool toStart = bool(msg.Get("to_start", false));
            if (toStart)
                sim.RequestFullRespawn();
            else
                sim.RequestRespawn();
        }
        else if (name == "prevent_finish")
        {
            g_PreventFinish = bool(msg.Get("value", true));
            if (g_PreventFinish)
                sim.PreventSimulationFinish();
            else
                sim.AllowSimulationFinish();
        }
    }

    void SendLog(const string &in text)
    {
        Json::Value msg = Json::Object();
        msg["type"] = "log";
        msg["message"] = text;
        string payload = Json::Write(msg) + "\n";
        array<uint8> data(payload.Length);
        for (uint i = 0; i < payload.Length; i++)
        {
            data[i] = payload[i];
        }
        if (g_Socket !is null && g_Socket.IsConnected())
            g_Socket.Send(data, data.Length);
    }
}
