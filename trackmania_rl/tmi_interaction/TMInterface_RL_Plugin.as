// TMInterface ≥2.0 AngelScript plugin used by TMNF-AI
// Copy this file into TMInterface/Plugins and enable it in the Plugins menu.
// Optional variables (set via TMInterface console or config):
//   rl_bridge_port         - TCP port used for the socket server (default 54540)
//   rl_bridge_timeout_ms   - Timeout in milliseconds for awaiting acknowledgements

Net::Socket@ gServer = null;
Net::Socket@ gClient = null;

const string HOST = "127.0.0.1";
uint16 PORT = 54540;
uint RESPONSE_TIMEOUT = 2000;

SimulationManager@ gSim = GetSimulationManager();
bool gPrevWallContact = false;
uint gWallContactCount = 0;

enum MessageType {
    SCRunStepSync = 1,
    SCCheckpointCountChangedSync = 2,
    SCLapCountChangedSync = 3,
    SCRequestedFrameSync = 4,
    SCOnConnectSync = 5,
    CSetSpeed = 6,
    CSetInputState = 10,
    CPreventSimulationFinish = 12,
    CShutdown = 13,
    CExecuteCommand = 14,
}

void Main() {
    RegisterVariable("rl_bridge_port", PORT);
    RegisterVariable("rl_bridge_timeout_ms", RESPONSE_TIMEOUT);
    PORT = uint16(GetVariableDouble("rl_bridge_port"));
    RESPONSE_TIMEOUT = uint(GetVariableDouble("rl_bridge_timeout_ms"));
    InitSocket();
}

void InitSocket() {
    if (@gServer !is null) {
        return;
    }
    @gServer = Net::Socket();
    gServer.Listen(HOST, PORT);
    log("RL bridge listening on " + HOST + ":" + PORT);
}

void CloseConnection() {
    @gClient = null;
    gPrevWallContact = false;
    gWallContactCount = 0;
    InitSocket();
}

void Render() {
    if (@gServer is null) {
        InitSocket();
    }

    auto@ newSock = gServer.Accept(0);
    if (@newSock !is null) {
        @gClient = newSock;
        gClient.NoDelay = true;
        gPrevWallContact = false;
        gWallContactCount = 0;
        log("RL bridge: client connected from " + gClient.RemoteIP);
        SendOnConnect();
    }

    while (HandleMessage() != -1) {
    }
}

void SendOnConnect() {
    if (@gClient is null) {
        return;
    }
    gClient.Write(MessageType::SCOnConnectSync);
    WaitForResponse(MessageType::SCOnConnectSync);
}

 void OnRunStep(SimulationManager@ sim) {
     if (@gClient is null || sim is null) {
         return;
     }

     @gSim = sim;

    while (HandleMessage() != -1) {
    }

     auto@ state = sim.SaveState();
    if (state is null) {
        return;
    }

    TM::HmsDyna@ dyna = state.Dyna;
    TM::SceneVehicleCar@ car = state.SceneVehicleCar;
    TM::PlayerInfo@ info = state.PlayerInfo;
    auto@ wheels = state.Wheels;

    if (dyna is null) {
        return;
    }

    // ВАЖНО: используем handle и RefStateCurrent
    TM::HmsStateDyna@ curState = dyna.RefStateCurrent;
    if (curState is null) {
    	return;
          }
    vec3 pos = curState.Location.Position;
    vec3 vel = curState.LinearSpeed;
    float speed = Math::Sqrt(vel.x * vel.x + vel.y * vel.y + vel.z * vel.z);

    quat q = curState.Quat;
    float yaw = 0.0f;
    float pitch = 0.0f;
    float roll = 0.0f;
    q.GetYawPitchRoll(yaw, pitch, roll);

    bool lateral = car !is null ? car.HasAnyLateralContact : false;
    if (lateral && !gPrevWallContact) {
        gWallContactCount++;
    }
    gPrevWallContact = lateral;

    int cpIndex = info is null ? 0 : info.CurCheckpoint;
    int lap = info is null ? 0 : info.CurLap;
    bool finished = info is null ? false : info.RaceFinished;

    bool fl = HasWheelGroundContact(wheels, 0);
    bool fr = HasWheelGroundContact(wheels, 1);
    bool rl = HasWheelGroundContact(wheels, 2);
    bool rr = HasWheelGroundContact(wheels, 3);
    uint grounded = (fl ? 1 : 0) + (fr ? 1 : 0) + (rl ? 1 : 0) + (rr ? 1 : 0);

    gClient.Write(MessageType::SCRunStepSync);
    gClient.Write(sim.RaceTime);
    gClient.Write(pos.x);
    gClient.Write(pos.y);
    gClient.Write(pos.z);
    gClient.Write(yaw);
    gClient.Write(speed);
    gClient.Write(vel.x);
    gClient.Write(vel.y);
    gClient.Write(vel.z);
    gClient.Write(cpIndex);
    gClient.Write(lap);
    gClient.Write(uint8(lateral ? 1 : 0));
    gClient.Write(uint8(finished ? 1 : 0));
    gClient.Write(uint8(fl ? 1 : 0));
    gClient.Write(uint8(fr ? 1 : 0));
    gClient.Write(uint8(rl ? 1 : 0));
    gClient.Write(uint8(rr ? 1 : 0));
    gClient.Write(uint8(grounded));
    gClient.Write(gWallContactCount);
    WaitForResponse(MessageType::SCRunStepSync);
}

bool HasWheelGroundContact(SimulationWheels@ wheels, uint index) {
    if (wheels is null) {
        return false;
    }
    TM::SceneVehicleCar::SimulationWheel@ wheel = null;
    if (index == 0) {
        @wheel = wheels.FrontLeft;
    } else if (index == 1) {
        @wheel = wheels.FrontRight;
    } else if (index == 2) {
        @wheel = wheels.BackLeft;
    } else if (index == 3) {
        @wheel = wheels.BackRight;
    }
    if (wheel is null) {
        return false;
    }
    return wheel.RTState.HasGroundContact;
}

void ApplyInputs(int steer, int gas, bool accelerate, bool brake) {
    if (@gSim is null) {
        return;
    }
    gSim.SetInputState(InputType::Steer, steer);
    gSim.SetInputState(InputType::Gas, gas);
    gSim.SetInputState(InputType::Up, accelerate ? 1 : 0);
    gSim.SetInputState(InputType::Down, brake ? 1 : 0);
}

void WaitForResponse(MessageType type) {
    auto start = Time::Now;
    while (true) {
        int receivedType = HandleMessage();
        if (receivedType == int(type)) {
            break;
        }
        if (receivedType == int(MessageType::CShutdown) || @gClient is null) {
            break;
        }
        if (receivedType == -1 && Time::Now - start > RESPONSE_TIMEOUT) {
            log("RL bridge: timeout waiting for response " + type);
            CloseConnection();
            break;
        }
    }
}

int HandleMessage() {
    if (@gClient is null) {
        return -1;
    }
    if (gClient.Available == 0) {
        return -1;
    }

    int type = gClient.ReadInt32();
    switch (type) {
        case MessageType::SCRunStepSync:
        case MessageType::SCCheckpointCountChangedSync:
        case MessageType::SCLapCountChangedSync:
        case MessageType::SCRequestedFrameSync:
        case MessageType::SCOnConnectSync: {
            break;
        }
        case MessageType::CSetSpeed: {
            float newSpeed = gClient.ReadFloat();
            if (@gSim !is null) {
                gSim.SetSpeed(newSpeed);
            }
            break;
        }
        case MessageType::CSetInputState: {
            int steer = gClient.ReadInt32();
            int gas = gClient.ReadInt32();
            bool accelerate = gClient.ReadUint8() > 0;
            bool brake = gClient.ReadUint8() > 0;
            ApplyInputs(steer, gas, accelerate, brake);
            break;
        }
        case MessageType::CPreventSimulationFinish: {
            if (@gSim !is null) {
                gSim.PreventSimulationFinish();
            }
            break;
        }
        case MessageType::CExecuteCommand: {
            int length = gClient.ReadInt32();
            string command = gClient.ReadString(length);
            ExecuteCommand(command);
            break;
        }
        case MessageType::CShutdown: {
            log("RL bridge: client disconnected");
            CloseConnection();
            break;
        }
        default: {
            log("RL bridge: unknown message " + type);
            break;
        }
    }

    return type;
}

PluginInfo@ GetPluginInfo() {
    PluginInfo info;
    info.Author = "TMNF-AI";
    info.Name = "RL Bridge";
    info.Description = "Socket bridge used by the python reinforcement learning client.";
    info.Version = "1.0";
    return info;
}
