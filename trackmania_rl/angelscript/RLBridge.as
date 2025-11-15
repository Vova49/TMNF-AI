// RLBridge.as
// -----------------------------------------------------------------------------
// AngelScript plugin that exposes TrackMania Nations Forever simulation data to
// an external reinforcement learning agent through a TCP socket.  The Python
// side of the project listens on ``cfg.tmiface.server_name:cfg.tmiface.bridge_port``
// and exchanges newline separated ``key=value`` payloads.
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
            g_ReadBuffer = "";
            SendLog("Connected to RL agent");
        }
    }

    void SendState(SimulationManager@ sim, TM::SceneVehicleCar@ car, TM::PlayerInfo@ player)
    {
        vec3 position = vec3();
        float yaw = 0.0f;

        TM::HmsDyna@ dyna = car.Dyna;
        vec3 velocity = vec3();
        if (dyna !is null)
        {
            TM::HmsStateDyna@ state = dyna.Forward;
            if (state !is null)
            {
                position = state.Location.Translation;
                velocity = state.LinearVelocity;
                yaw = Math::Atan2(state.Location.XX, state.Location.ZX);
            }
        }

        float speed = 0.0f;
        float raceTime = 0.0f;
        int checkpoint = 0;
        bool raceFinished = false;
        bool hasLateralContact = car.HasAnyLateralContact;

        if (player !is null)
        {
            speed = player.DisplaySpeed;
            raceTime = float(player.RaceTime);
            checkpoint = player.CurCheckpoint;
            raceFinished = player.RaceFinished;
        }
        else if (velocity.LengthSquared() > 0.0f)
        {
            speed = velocity.Length();
        }

        array<string> parts;
        parts.InsertLast("type=state");
        parts.InsertLast("speed=" + FormatFloat(speed));
        parts.InsertLast("yaw=" + FormatFloat(yaw));
        parts.InsertLast("race_time=" + FormatFloat(raceTime));
        parts.InsertLast("cp_index=" + FormatInt(checkpoint));
        parts.InsertLast("race_finished=" + (raceFinished ? "1" : "0"));
        parts.InsertLast("has_any_lateral_contact=" + (hasLateralContact ? "1" : "0"));
        parts.InsertLast("position=" + EncodeVec3(position));
        parts.InsertLast("velocity=" + EncodeVec3(velocity));
        parts.InsertLast("wheel_ground_contact=" + EncodeWheelContacts(car));
        parts.InsertLast("nb_wheels_grounded=" + FormatInt(int(car.NbWheelsGrounded)));
        parts.InsertLast("just_respawned=" + (sim.PlayerJustRespawned ? "1" : "0"));

        string payload = JoinParts(parts, ";") + "\n";
        SendString(payload);
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

            dictionary@ msg = ParseMessage(line);
            if (msg is null)
                continue;

            string msgType = GetString(msg, "type", "inputs");
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

    void ApplyInputs(TM::SceneVehicleCar@ car, dictionary@ msg)
    {
        g_LastSteer = GetFloat(msg, "steer", g_LastSteer);
        g_LastGas = GetFloat(msg, "gas", g_LastGas);
        g_LastBrake = GetBool(msg, "brake", g_LastBrake);

        car.InputSteer = Math::Clamp(g_LastSteer, -1.0f, 1.0f);
        car.InputGasPedal = Math::Clamp(g_LastGas, 0.0f, 1.0f);
        car.InputBrake = g_LastBrake;
    }

    void HandleCommand(SimulationManager@ sim, dictionary@ msg)
    {
        string name = GetString(msg, "name", "");
        if (name == "set_game_speed")
        {
            float speed = GetFloat(msg, "value", 1.0f);
            sim.GameSpeed = speed;
        }
        else if (name == "respawn")
        {
            bool toStart = GetBool(msg, "to_start", false);
            if (toStart)
                sim.RequestFullRespawn();
            else
                sim.RequestRespawn();
        }
        else if (name == "prevent_finish")
        {
            g_PreventFinish = GetBool(msg, "value", true);
            if (g_PreventFinish)
                sim.PreventSimulationFinish();
            else
                sim.AllowSimulationFinish();
        }
    }

    void SendLog(const string &in text)
    {
        string sanitized = text;
        sanitized.Replace(";", ",");
        sanitized.Replace("=", "-");
        array<string> parts;
        parts.InsertLast("type=log");
        parts.InsertLast("message=" + sanitized);
        SendString(JoinParts(parts, ";") + "\n");
    }

    void SendString(const string &in payload)
    {
        if (g_Socket is null || !g_Socket.IsConnected())
            return;

        array<uint8> data(payload.Length);
        for (uint i = 0; i < payload.Length; i++)
        {
            data[i] = payload[i];
        }
        g_Socket.Send(data, data.Length);
    }

    string JoinParts(array<string>@ parts, const string &in separator)
    {
        string result;
        for (uint i = 0; i < parts.Length; i++)
        {
            if (i > 0)
                result += separator;
            result += parts[i];
        }
        return result;
    }

    string EncodeVec3(const vec3 &in value)
    {
        return FormatFloat(value.x) + "," + FormatFloat(value.y) + "," + FormatFloat(value.z);
    }

    string EncodeWheelContacts(TM::SceneVehicleCar@ car)
    {
        array<string> flags;
        for (uint i = 0; i < car.Wheels.Length; i++)
        {
            auto@ wheel = car.Wheels[i];
            bool grounded = wheel !is null && wheel.RealTimeState.HasGroundContact;
            flags.InsertLast(grounded ? "1" : "0");
        }
        return JoinParts(flags, ",");
    }

    string FormatFloat(float value)
    {
        return Text::Format("%.6f", value);
    }

    string FormatInt(int value)
    {
        return Text::Format("%d", value);
    }

    dictionary@ ParseMessage(const string &in line)
    {
        dictionary@ msg = dictionary();
        int start = 0;
        int total = int(line.Length);
        while (start < total)
        {
            int sep = line.IndexOf(';', start);
            if (sep < 0)
                sep = total;

            string token = line.SubStr(start, sep - start);
            int eq = token.IndexOf('=');
            if (eq >= 0)
            {
                string key = token.SubStr(0, eq);
                string value = token.SubStr(eq + 1);
                msg.Set(key, value);
            }

            start = sep + 1;
        }
        return msg;
    }

    string GetString(dictionary@ msg, const string &in key, const string &in defaultValue)
    {
        string value;
        if (msg.Get(key, value))
            return value;
        return defaultValue;
    }

    float GetFloat(dictionary@ msg, const string &in key, float defaultValue)
    {
        string raw;
        if (!msg.Get(key, raw) || raw.Length == 0)
            return defaultValue;
        float parsed = Text::ParseFloat(raw);
        if (Math::IsNaN(parsed))
            return defaultValue;
        return parsed;
    }

    bool GetBool(dictionary@ msg, const string &in key, bool defaultValue)
    {
        string raw;
        if (!msg.Get(key, raw) || raw.Length == 0)
            return defaultValue;
        string lowered = Text::ToLower(raw);
        if (lowered == "1" || lowered == "true" || lowered == "yes")
            return true;
        if (lowered == "0" || lowered == "false" || lowered == "no")
            return false;
        return defaultValue;
    }
}
