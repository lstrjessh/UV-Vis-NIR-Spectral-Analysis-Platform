
Imports System.Drawing
Imports System.Runtime.InteropServices
Imports System.Text
Imports System.Runtime.InteropServices.ComTypes

#If ALLOW_UNTESTED_INTERFACES Then

Public Enum VideoCopyProtectionType
    MacrovisionBasic
    MacrovisionCBI
End Enum

<Flags()> _
Public Enum AMPushSourceFlags
    None = 0
    InternalRM = &H1
    NotLive = &H2
    PrivateClock = &H4
    UseStreamClock = &H10000
    UseClockChain = &H20000
End Enum

Public Enum DVResolution
    Full = 1000
    Half = 1001
    Quarter = 1002
    Dc = 1003
End Enum

Public Enum VideoEncoderBitrateMode
    ConstantBitRate = 0
    VariableBitRateAverage
    VariableBitRatePeak
End Enum

<Flags()> _
Public Enum RegPinFlag
    None = 0
    Zero = &H1
    Renderer = &H2
    Many = &H4
    Output = &H8
End Enum

<Flags()> _
Public Enum Advise
    None = &H0
    Clipping = &H1
    Palette = &H2
    ColorKey = &H4
    Position = &H8
    DisplayChange = &H10
    All = Advise.Clipping Or Advise.Palette Or Advise.ColorKey Or Advise.Position
    All2 = Advise.All Or Advise.DisplayChange
End Enum

<StructLayout(LayoutKind.Sequential)> _
Public Structure RegFilter
    Public Clsid As Guid
    <MarshalAs(UnmanagedType.LPWStr)> _
    Public Name As String
End Structure

<StructLayout(LayoutKind.Sequential)> _
Public Structure RegPinTypes
    Public clsMajorType As Guid
    Public clsMinorType As Guid
End Structure

<StructLayout(LayoutKind.Sequential)> _
Public Structure RegFilterPins
    <MarshalAs(UnmanagedType.LPWStr)> _
    Public strName As String
    <MarshalAs(UnmanagedType.Bool)> _
    Public bRendered As Boolean
    <MarshalAs(UnmanagedType.Bool)> _
    Public bOutput As Boolean
    <MarshalAs(UnmanagedType.Bool)> _
    Public bZero As Boolean
    <MarshalAs(UnmanagedType.Bool)> _
    Public bMany As Boolean
    Public clsConnectsToFilter As Guid
    <MarshalAs(UnmanagedType.LPWStr)> _
    Public strConnectsToPin As String
    Public nMediaTypes As Integer
    <MarshalAs(UnmanagedType.ByValArray)> _
    Public lpMediaType As RegPinTypes()
End Structure

<StructLayout(LayoutKind.Sequential)> _
Public Structure RegFilterPins2
    Public dwFlags As RegPinFlag
    Public cInstances As Integer
    Public nMediaTypes As Integer
    Public lpMediaType As RegPinTypes()
    Public nMediums As Integer
    Public lpMedium As RegPinMedium()
    Public clsPinCategory As Guid
End Structure

<StructLayout(LayoutKind.Explicit)> _
Public Structure RegFilter2Union
    <FieldOffset(0)> _
    Public rgPins As RegFilterPins()
    <FieldOffset(0)> _
    Public rgPins2 As RegFilterPins2()
End Structure

<StructLayout(LayoutKind.Sequential)> _
Public Structure RegFilter2
    Public dwVersion As Integer
    Public dwMerit As Integer
    Public cPins As Integer
    Public rgPins As RegFilter2Union
End Structure

<StructLayout(LayoutKind.Sequential)> _
Public Structure RgnDataHeader
    Public dwSize As Integer
    Public iType As Integer
    Public nCount As Integer
    Public nRgnSize As Integer
    Public rcBound As Rectangle
End Structure

<StructLayout(LayoutKind.Sequential)> _
Public Structure RgnData
    Public rdh As RgnDataHeader
    Public Buffer As IntPtr
End Structure

<StructLayout(LayoutKind.Sequential, Pack:=1)> _
Public Structure TimeCode
    Public wFrameRate As Short
    Public wFrameFract As Short
    Public dwFrames As Integer
End Structure

<StructLayout(LayoutKind.Sequential)> _
Public Structure TimeCodeSample
    Public qwTick As Long
    Public timecode As TimeCode
    Public dwUser As Integer
    Public dwFlags As Integer
End Structure

<StructLayout(LayoutKind.Sequential)> _
Public Structure StreamIdMap
    Public stream_id As Integer
    Public dwMediaSampleContent As Integer
    Public ulSubstreamFilterValue As Integer
    Public iDataOffset As Integer
End Structure

<StructLayout(LayoutKind.Sequential)> _
Public Structure CodecAPIEventData
    Public guid As Guid
    Public dataLength As Integer
    Public reserved1 As Integer
    Public reserved2 As Integer
    Public reserved3 As Integer
End Structure

<StructLayout(LayoutKind.Sequential)> _
Public Structure AMCOPPSignature
    <MarshalAs(UnmanagedType.ByValArray, ArraySubType:=UnmanagedType.I1, SizeConst:=256)> _
    Public Signature As Byte()
End Structure

<StructLayout(LayoutKind.Sequential)> _
Public Class AMCOPPCommand
    Public macKDI As Guid
    Public guidCommandID As Guid
    Public dwSequence As Integer
    Public cbSizeData As Integer
    <MarshalAs(UnmanagedType.ByValArray, ArraySubType:=UnmanagedType.I1, SizeConst:=4056)> _
    Public CommandData As Byte()
End Class

<StructLayout(LayoutKind.Sequential)> _
Public Class AMCOPPStatusInput
    Public rApp As Guid
    Public guidStatusRequestID As Guid
    Public dwSequence As Integer
    Public cbSizeData As Integer
    <MarshalAs(UnmanagedType.ByValArray, ArraySubType:=UnmanagedType.I1, SizeConst:=4056)> _
    Public StatusData As Byte()
End Class

<StructLayout(LayoutKind.Sequential)> _
Public Structure AMCOPPStatusOutput
    Public macKDI As Guid
    Public cbSizeData As Integer
    <MarshalAs(UnmanagedType.ByValArray, ArraySubType:=UnmanagedType.I1, SizeConst:=4076)> _
    Public COPPStatus As Byte()
End Structure

#End If


<Flags()> _
Public Enum AMFilterMiscFlags
    None = 0
    IsRenderer = &H1
    IsSource = &H2
End Enum

<Flags()> _
Public Enum AMStreamInfoFlags
    None = &H0
    StartDefined = &H1
    StopDefined = &H2
    Discarding = &H4
    StopSendExtra = &H10
End Enum

Public Enum MPEG2Program
    StreamMap = &H0
    ElementaryStream = &H1
    DirecoryPesPacket = &H2
    PackHeader = &H3
    PesSteam = &H4
    SystemHeader = &H5
End Enum

Public Enum AMAudioRendererStatParam
    BreakCount = 1
    SlaveMode
    SilenceDur
    LastBufferDur
    Discontinuities
    SlaveRate
    SlaveDropWriteDur
    SlaveHighLowError
    SlaveLastHighLowError
    SlaveAccumError
    BufferFullness
    Jitter
End Enum

<StructLayout(LayoutKind.Sequential)> _
Public Structure AMStreamInfo
    Public tStart As Long
    Public tStop As Long
    Public dwStartCookie As Integer
    Public dwStopCookie As Integer
    Public dwFlags As AMStreamInfoFlags
End Structure

Public Enum DVDecoderResolution
    r720x480 = 1000
    r360x240 = 1001
    r180x120 = 1002
    r88x60 = 1003
End Enum

<Flags()> _
Public Enum AMIntfSearchFlags
    None = &H0
    InputPin = &H1
    OutputPin = &H2
    Filter = &H4
End Enum

Public Enum AMQueryDecoder
    VMRSupport = &H1
    DXVA_1Support = &H2
    DVDSupport = &H3
    ATSC_SDSupport = &H4
    ATSC_HDSupport = &H5
    VMR9Support = &H6
End Enum

Public Enum DecoderCap
    NotSupported = &H0
    Supported = &H1
End Enum

Public Enum DecimationUsage
    Legacy
    UseDecoderOnly
    UseVideoPortOnly
    UseOverlayOnly
    [Default]
End Enum

<Flags()> _
Public Enum AMOverlayFX
    NoFX = &H0
    MirrorLeftRight = &H2
    MirrorUpDown = &H4
    Deinterlace = &H8
End Enum

<Flags()> _
Public Enum AMResCtlReserveFlags
    Reserve = &H0
    UnReserve = &H1
End Enum

<Flags()> _
Public Enum AMStreamSelectInfoFlags
    Disabled = &H0
    Enabled = &H1
    Exclusive = &H2
End Enum

<Flags()> _
Public Enum AMStreamSelectEnableFlags
    DisableAll = &H0
    Enable = &H1
    EnableAll = &H2
End Enum

<Flags()> _
Public Enum Merit
    None = 0
    Preferred = &H800000
    Normal = &H600000
    Unlikely = &H400000
    DoNotUse = &H200000
    SWCompressor = &H100000
    HWCompressor = &H100050
End Enum

<StructLayout(LayoutKind.Sequential)> _
Public Class ColorKey
    Public KeyType As Integer
    Public PaletteIndex As Integer
    Public LowColorValue As Integer
    Public HighColorValue As Integer
End Class

<StructLayout(LayoutKind.Sequential)> _
Public Class RegPinMedium
    Public clsMedium As Guid
    Public dw1 As Integer
    Public dw2 As Integer
End Class

<StructLayout(LayoutKind.Sequential)> _
Public Structure DVInfo
    Public dwDVAAuxSrc As Integer
    Public dwDVAAuxCtl As Integer
    Public dwDVAAuxSrc1 As Integer
    Public dwDVAAuxCtl1 As Integer
    Public dwDVVAuxSrc As Integer
    Public dwDVVAuxCtl As Integer
    Public dwDVReserved1 As Integer
    Public dwDVReserved2 As Integer
End Structure

Public Enum DVEncoderResolution
    r720x480 = 2012
    r360x240 = 2013
    r180x120 = 2014
    r88x60 = 2015
End Enum

Public Enum DVEncoderFormat
    DVSD = 2007
    DVHD = 2008
    DVSL = 2009
End Enum

Public Enum DVEncoderVideoFormat
    NTSC = 2000
    PAL = 2001
End Enum

<Flags()> _
Public Enum AMRenderExFlags
    None = 0
    RenderToExistingRenderers = 1
End Enum

Public Enum InterleavingMode
    None
    Capture
    Full
    NoneBuffered
End Enum

<Flags()> _
Public Enum AMFileSinkFlags
    None = 0
    OverWrite = &H1
End Enum

Public Enum KSPropertySupport
    [Get] = 1
    [Set] = 2
End Enum

Public Enum AMPropertyPin
    Category
    Medium
End Enum

Public Enum AMTunerSubChannel
    NoTune = -2
    [Default] = -1
End Enum

Public Enum AMTunerSignalStrength
    HasNoSignalStrength = -1
    NoSignal = 0
    SignalPresent = 1
End Enum

<Flags()> _
Public Enum AMTunerModeType
    [Default] = &H0
    TV = &H1
    FMRadio = &H2
    AMRadio = &H4
    Dss = &H8
    DTV = &H10
End Enum

Public Enum AMTunerEventType
    Changed = &H1
End Enum

<Flags()> _
Public Enum AnalogVideoStandard
    None = &H0
    NTSC_M = &H1
    NTSC_M_J = &H2
    NTSC_433 = &H4
    PAL_B = &H10
    PAL_D = &H20
    PAL_G = &H40
    PAL_H = &H80
    PAL_I = &H100
    PAL_M = &H200
    PAL_N = &H400
    PAL_60 = &H800
    SECAM_B = &H1000
    SECAM_D = &H2000
    SECAM_G = &H4000
    SECAM_H = &H8000
    SECAM_K = &H10000
    SECAM_K1 = &H20000
    SECAM_L = &H40000
    SECAM_L1 = &H80000
    PAL_N_COMBO = &H100000

    NTSCMask = &H7
    PALMask = &H100FF0
    SECAMMask = &HFF000
End Enum

Public Enum TunerInputType
    Cable
    Antenna
End Enum

<Flags()> _
Public Enum VideoControlFlags
    None = &H0
    FlipHorizontal = &H1
    FlipVertical = &H2
    ExternalTriggerEnable = &H4
    Trigger = &H8
End Enum

<Flags()> _
Public Enum TVAudioMode
    None = 0
    Mono = &H1
    Stereo = &H2
    LangA = &H10
    LangB = &H20
    LangC = &H40
End Enum

Public Enum VideoProcAmpProperty
    Brightness
    Contrast
    Hue
    Saturation
    Sharpness
    Gamma
    ColorEnable
    WhiteBalance
    BacklightCompensation
    Gain
End Enum

<Flags()> _
Public Enum VideoProcAmpFlags
    None = 0
    Auto = &H1
    Manual = &H2
End Enum

Public Enum PhysicalConnectorType
    Video_Tuner = 1
    Video_Composite
    Video_SVideo
    Video_RGB
    Video_YRYBY
    Video_SerialDigital
    Video_ParallelDigital
    Video_SCSI
    Video_AUX
    Video_1394
    Video_USB
    Video_VideoDecoder
    Video_VideoEncoder
    Video_SCART
    Video_Black

    Audio_Tuner = &H1000
    Audio_Line
    Audio_Mic
    Audio_AESDigital
    Audio_SPDIFDigital
    Audio_SCSI
    Audio_AUX
    Audio_1394
    Audio_USB
    Audio_AudioDecoder
End Enum

<Flags()> _
Public Enum AMTVAudioEventType
    None = 0
    Changed = &H1
End Enum

<Flags()> _
Public Enum CompressionCaps
    None = &H0
    CanQuality = &H1
    CanCrunch = &H2
    CanKeyFrame = &H4
    CanBFrame = &H8
    CanWindow = &H10
End Enum

<Flags()> _
Public Enum VfwCompressDialogs
    None = 0
    Config = &H1
    About = &H2
    QueryConfig = &H4
    QueryAbout = &H8
End Enum

<Flags()> _
Public Enum VfwCaptureDialogs
    None = &H0
    Source = &H1
    Format = &H2
    Display = &H4
End Enum

Public Enum ExtDevicePort
    Sim = 1
    Com1 = 2
    Com2 = 3
    Com3 = 4
    Com4 = 5
    Diaq = 6
    Arti = 7
    FireWire1394 = 8
    Usb = 9
    Min = Sim
    Max = Usb
End Enum

Public Enum ExtDeviceBase
    Base = &H1000
End Enum

Public Enum ExtDeviceCaps
    None = 0
    CanRecord = ExtDeviceBase.Base + 1
    CanRecordStrobe = ExtDeviceBase.Base + 2
    HasAudio = ExtDeviceBase.Base + 3
    HasVideo = ExtDeviceBase.Base + 4
    UsesFiles = ExtDeviceBase.Base + 5
    CanSave = ExtDeviceBase.Base + 6
    DeviceType = ExtDeviceBase.Base + 7
    VCR = ExtDeviceBase.Base + 8
    LaserDisk = ExtDeviceBase.Base + 9
    ATR = ExtDeviceBase.Base + 10
    DDR = ExtDeviceBase.Base + 11
    Router = ExtDeviceBase.Base + 12
    Keyer = ExtDeviceBase.Base + 13
    MixerVideo = ExtDeviceBase.Base + 14
    DVE = ExtDeviceBase.Base + 15
    WipeGen = ExtDeviceBase.Base + 16
    MixerAudio = ExtDeviceBase.Base + 17
    CG = ExtDeviceBase.Base + 18
    TBC = ExtDeviceBase.Base + 19
    TCG = ExtDeviceBase.Base + 20
    GPI = ExtDeviceBase.Base + 21
    Joystick = ExtDeviceBase.Base + 22
    Keyboard = ExtDeviceBase.Base + 3
    ExternalDeviceID = ExtDeviceBase.Base + 24
    TimeCodeRead = ExtDeviceBase.Base + 25
    TimeCodeWrite = ExtDeviceBase.Base + 26
    CtlTrkRead = ExtDeviceBase.Base + 27
    IndexRead = ExtDeviceBase.Base + 28
    PreRoll = ExtDeviceBase.Base + 29
    PostRoll = ExtDeviceBase.Base + 30
    SyncAccuracy = ExtDeviceBase.Base + 31
    Precise = ExtDeviceBase.Base + 32
    Frame = ExtDeviceBase.Base + 33
    Rough = ExtDeviceBase.Base + 34
    NormalRate = ExtDeviceBase.Base + 35
    Rate24 = ExtDeviceBase.Base + 36
    Rate25 = ExtDeviceBase.Base + 37
    Rate2997 = ExtDeviceBase.Base + 38
    Rate30 = ExtDeviceBase.Base + 39
    CanPreview = ExtDeviceBase.Base + 40
    CanMonitorSources = ExtDeviceBase.Base + 41
    CanTest = ExtDeviceBase.Base + 42
    VideoInputs = ExtDeviceBase.Base + 43
    AudioInputs = ExtDeviceBase.Base + 44
    NeedsCalibrating = ExtDeviceBase.Base + 45
    SeekType = ExtDeviceBase.Base + 46
    Perfect = ExtDeviceBase.Base + 47
    Fast = ExtDeviceBase.Base + 48
    Slow = ExtDeviceBase.Base + 49
    [On] = ExtDeviceBase.Base + 50
    Off = ExtDeviceBase.Base + 51
    Standby = ExtDeviceBase.Base + 52
    All = ExtDeviceBase.Base + 55
    Test = ExtDeviceBase.Base + 56
    DeviceTypeCamera = ExtDeviceBase.Base + 900
    DeviceTypeTuner = ExtDeviceBase.Base + 901
    DeviceTypeDvhs = ExtDeviceBase.Base + 902
    DeviceTypeUnknown = ExtDeviceBase.Base + 903
    CapabilityUnknown = ExtDeviceBase.Base + 910
End Enum

Public Enum ExtTransportCaps
    None = 0
    CanEject = ExtDeviceBase.Base + 100
    CanBumpPlay = ExtDeviceBase.Base + 101
    CanPlayBackwards = ExtDeviceBase.Base + 102
    CanSetEE = ExtDeviceBase.Base + 103
    CanSetPB = ExtDeviceBase.Base + 104
    CanDelayVideoIn = ExtDeviceBase.Base + 105
    CanDelayVideoOut = ExtDeviceBase.Base + 106
    CanDelayAudioIn = ExtDeviceBase.Base + 107
    CanDelayAudioOut = ExtDeviceBase.Base + 108
    FwdVariableMax = ExtDeviceBase.Base + 109
    FwdVariableMin = ExtDeviceBase.Base + 800
    RevVariableMax = ExtDeviceBase.Base + 110
    RevVariableMin = ExtDeviceBase.Base + 801
    FwdShuttleMax = ExtDeviceBase.Base + 802
    FwdShuttleMin = ExtDeviceBase.Base + 803
    RevShuttleMax = ExtDeviceBase.Base + 804
    RevShuttleMin = ExtDeviceBase.Base + 805
    NumAudioTracks = ExtDeviceBase.Base + 111
    LTCTrack = ExtDeviceBase.Base + 112
    NeedsTBC = ExtDeviceBase.Base + 113
    NeedsCueing = ExtDeviceBase.Base + 114
    CanInsert = ExtDeviceBase.Base + 115
    CanAssemble = ExtDeviceBase.Base + 116
    FieldStep = ExtDeviceBase.Base + 117
    ClockIncRate = ExtDeviceBase.Base + 118
    CanDetechLength = ExtDeviceBase.Base + 119
    CanFreeze = ExtDeviceBase.Base + 120
    HasTuner = ExtDeviceBase.Base + 121
    HasTimer = ExtDeviceBase.Base + 122
    HasClock = ExtDeviceBase.Base + 123
    MultipleEdits = ExtDeviceBase.Base + 806
    IsMaster = ExtDeviceBase.Base + 807
    HasDT = ExtDeviceBase.Base + 814
End Enum

Public Enum ExtTransportMediaStates
    None = 0
    SpinUp = ExtDeviceBase.Base + 130
    SpinDown = ExtDeviceBase.Base + 131
    Unload = ExtDeviceBase.Base + 132
End Enum

Public Enum ExtTransportModes
    None = 0
    Play = ExtDeviceBase.Base + 200
    [Stop] = ExtDeviceBase.Base + 201
    Freeze = ExtDeviceBase.Base + 202
    Thaw = ExtDeviceBase.Base + 203
    FF = ExtDeviceBase.Base + 204
    Rew = ExtDeviceBase.Base + 205
    Record = ExtDeviceBase.Base + 206
    RecordStrobe = ExtDeviceBase.Base + 207
    RecordFreeze = ExtDeviceBase.Base + 808
    [Step] = ExtDeviceBase.Base + 208
    StepFwd = [Step]
    StepRew = ExtDeviceBase.Base + 809
    Shuttle = ExtDeviceBase.Base + 209
    EditCue = ExtDeviceBase.Base + 210
    VarSpeed = ExtDeviceBase.Base + 211
    Perform = ExtDeviceBase.Base + 212
    LinkOn = ExtDeviceBase.Base + 280
    LinkOff = ExtDeviceBase.Base + 281
    NotifyEnable = ExtDeviceBase.Base + 810
    NotifyDisable = ExtDeviceBase.Base + 811
    ShotSearch = ExtDeviceBase.Base + 812
    PlayFastestFwd = ExtDeviceBase.Base + 933
    PlaySlowestFwd = ExtDeviceBase.Base + 934
    PlayFastestRev = ExtDeviceBase.Base + 935
    PlaySlowestRev = ExtDeviceBase.Base + 936
    Wind = ExtDeviceBase.Base + 937
    RewFastest = ExtDeviceBase.Base + 938
    RevPlay = ExtDeviceBase.Base + 939
End Enum

Public Enum ExtTransportStatus
    None = 0
    Mode = ExtDeviceBase.Base + 500
    [Error] = ExtDeviceBase.Base + 501
    Local = ExtDeviceBase.Base + 502
    RecordInhibit = ExtDeviceBase.Base + 503
    ServoLock = ExtDeviceBase.Base + 504
    MediaPresent = ExtDeviceBase.Base + 505
    MediaLength = ExtDeviceBase.Base + 506
    MediaSize = ExtDeviceBase.Base + 507
    MediaTrackCount = ExtDeviceBase.Base + 508
    MediaTrackLength = ExtDeviceBase.Base + 509
    MediaSide = ExtDeviceBase.Base + 510
    MediaType = ExtDeviceBase.Base + 511
    MediaVhs = ExtDeviceBase.Base + 512
    MediaSvhs = ExtDeviceBase.Base + 513
    MediaHi8 = ExtDeviceBase.Base + 514
    MediaUmatic = ExtDeviceBase.Base + 515
    MediaDvc = ExtDeviceBase.Base + 516
    Media1Inch = ExtDeviceBase.Base + 517
    MediaD1 = ExtDeviceBase.Base + 518
    MediaD2 = ExtDeviceBase.Base + 519
    MediaD3 = ExtDeviceBase.Base + 520
    MediaD5 = ExtDeviceBase.Base + 521
    MediaDBeta = ExtDeviceBase.Base + 522
    MediaBeta = ExtDeviceBase.Base + 523
    Media8mm = ExtDeviceBase.Base + 524
    MediaDdr = ExtDeviceBase.Base + 525
    MediaSx = ExtDeviceBase.Base + 813
    MediaOther = ExtDeviceBase.Base + 526
    MediaClv = ExtDeviceBase.Base + 527
    MediaCav = ExtDeviceBase.Base + 528
    MediaPosition = ExtDeviceBase.Base + 529
    MediaNeo = ExtDeviceBase.Base + 531
    MediaVhsc = ExtDeviceBase.Base + 925
    MediaUnknown = ExtDeviceBase.Base + 926
    MediaNotPresent = ExtDeviceBase.Base + 927
    LinkMode = ExtDeviceBase.Base + 530
    DevRemovedHeventGet = ExtDeviceBase.Base + 960
    DevRemovedHeventRelease = ExtDeviceBase.Base + 961
    ModeChangeNotify = ExtDeviceBase.Base + 932
    ControlHeventGet = ExtDeviceBase.Base + 928
    ControlHeventRelease = ExtDeviceBase.Base + 929
    NotifyHeventGet = ExtDeviceBase.Base + 930
    NotifyHeventRelease = ExtDeviceBase.Base + 931
End Enum

Public Enum ExtTransportParameters
    None = 0
    TimeFormat = ExtDeviceBase.Base + 540
    TimeFormatMilliseconds = ExtDeviceBase.Base + 541
    TimeFormatFrames = ExtDeviceBase.Base + 542
    TimeFormatReferenceTime = ExtDeviceBase.Base + 543
    TimeFormatHmsf = ExtDeviceBase.Base + 547
    TimeFormatTmsf = ExtDeviceBase.Base + 548
    TimeReference = ExtDeviceBase.Base + 549
    TimeReferenceTimeCode = ExtDeviceBase.Base + 550
    TimeReferenceControlTrack = ExtDeviceBase.Base + 551
    TimeReferenceIndex = ExtDeviceBase.Base + 552
    TimeReferenceAtn = ExtDeviceBase.Base + 958
    SuperImpose = ExtDeviceBase.Base + 553
    EndStopAction = ExtDeviceBase.Base + 554
    RecordFormat = ExtDeviceBase.Base + 555
    RecordFormatSp = ExtDeviceBase.Base + 556
    RecordFormatLp = ExtDeviceBase.Base + 557
    RecordFormatEp = ExtDeviceBase.Base + 558
    StepCount = ExtDeviceBase.Base + 559
    StepUnit = ExtDeviceBase.Base + 560
    StepField = ExtDeviceBase.Base + 561
    StepFrame = ExtDeviceBase.Base + 562
    Step3_2 = ExtDeviceBase.Base + 563
    PreRoll = ExtDeviceBase.Base + 564
    RecPreRoll = ExtDeviceBase.Base + 565
    PostRoll = ExtDeviceBase.Base + 566
    EditDelay = ExtDeviceBase.Base + 567
    PlayTcDelay = ExtDeviceBase.Base + 568
    RecTcDelay = ExtDeviceBase.Base + 569
    EditField = ExtDeviceBase.Base + 570
    FrameServo = ExtDeviceBase.Base + 571
    CfServo = ExtDeviceBase.Base + 572
    ServoRef = ExtDeviceBase.Base + 573
    ServoRefExternal = ExtDeviceBase.Base + 574
    ServoRefInput = ExtDeviceBase.Base + 575
    ServoRefInternal = ExtDeviceBase.Base + 576
    ServoRefAuto = ExtDeviceBase.Base + 577
    WarnGl = ExtDeviceBase.Base + 578
    SetTracking = ExtDeviceBase.Base + 579
    SetTrackingPlus = ExtDeviceBase.Base + 580
    SetTrackingMinus = ExtDeviceBase.Base + 581
    SetTrackingReset = ExtDeviceBase.Base + 582
    SetFreezeTimeout = ExtDeviceBase.Base + 583
    VolumeName = ExtDeviceBase.Base + 584
    Ballistic_1 = ExtDeviceBase.Base + 585
    Ballistic_2 = ExtDeviceBase.Base + 586
    Ballistic_3 = ExtDeviceBase.Base + 587
    Ballistic_4 = ExtDeviceBase.Base + 588
    Ballistic_5 = ExtDeviceBase.Base + 589
    Ballistic_6 = ExtDeviceBase.Base + 590
    Ballistic_7 = ExtDeviceBase.Base + 591
    Ballistic_8 = ExtDeviceBase.Base + 592
    Ballistic_9 = ExtDeviceBase.Base + 593
    Ballistic_10 = ExtDeviceBase.Base + 594
    Ballistic_11 = ExtDeviceBase.Base + 595
    Ballistic_12 = ExtDeviceBase.Base + 596
    Ballistic_13 = ExtDeviceBase.Base + 597
    Ballistic_14 = ExtDeviceBase.Base + 598
    Ballistic_15 = ExtDeviceBase.Base + 599
    Ballistic_16 = ExtDeviceBase.Base + 600
    Ballistic_17 = ExtDeviceBase.Base + 601
    Ballistic_18 = ExtDeviceBase.Base + 602
    Ballistic_19 = ExtDeviceBase.Base + 603
    Ballistic_20 = ExtDeviceBase.Base + 604
    SetClock = ExtDeviceBase.Base + 605
    SetCounterFormat = ExtDeviceBase.Base + 606
    SetCounterValue = ExtDeviceBase.Base + 607
    SetTunerChUp = ExtDeviceBase.Base + 608
    SetTunerChDn = ExtDeviceBase.Base + 609
    SetTunerSkUp = ExtDeviceBase.Base + 610
    SetTunerSkDn = ExtDeviceBase.Base + 611
    SetTunerCh = ExtDeviceBase.Base + 612
    SetTunerNum = ExtDeviceBase.Base + 613
    SetTimerEvent = ExtDeviceBase.Base + 614
    SetTimerStartDay = ExtDeviceBase.Base + 615
    SetTimerStartTime = ExtDeviceBase.Base + 616
    SetTimerStopDay = ExtDeviceBase.Base + 617
    SetTimerStopTime = ExtDeviceBase.Base + 618
    VideoSetOutput = ExtDeviceBase.Base + 630
    E2E = ExtDeviceBase.Base + 631
    Playback = ExtDeviceBase.Base + 632
    Off = ExtDeviceBase.Base + 633
    VideoSetSource = ExtDeviceBase.Base + 634
    AudioEnableOutput = ExtDeviceBase.Base + 640
    AudioEnableRecord = ExtDeviceBase.Base + 642
    AudioEnableSelsync = ExtDeviceBase.Base + 643
    AudioSetSource = ExtDeviceBase.Base + 644
    AudioSetMonitor = ExtDeviceBase.Base + 645
    RawExtDeviceCommand = ExtDeviceBase.Base + 920
    InputSignal = ExtDeviceBase.Base + 940
    OutputSignal = ExtDeviceBase.Base + 941
    Signal_525_60_SD = ExtDeviceBase.Base + 942
    Signal_525_60_SDL = ExtDeviceBase.Base + 943
    Signal_625_50_SD = ExtDeviceBase.Base + 944
    Signal_625_50_SDL = ExtDeviceBase.Base + 945
    Signal_MPEG2TS = ExtDeviceBase.Base + 946
    Signal_625_60_HD = ExtDeviceBase.Base + 947
    Signal_625_50_HD = ExtDeviceBase.Base + 948
    Signal_2500_60_MPEG = ExtDeviceBase.Base + 980
    Signal_1250_60_MPEG = ExtDeviceBase.Base + 981
    Signal_0625_60_MPEG = ExtDeviceBase.Base + 982
    Signal_2500_50_MPEG = ExtDeviceBase.Base + 985
    Signal_1250_50_MPEG = ExtDeviceBase.Base + 986
    Signal_0625_50_MPEG = ExtDeviceBase.Base + 987
    SignalUnknown = ExtDeviceBase.Base + 990
End Enum

<Flags()> _
Public Enum ExtTransportAudio
    None = 0
    AudioAll = &H10000000
    ' -------------------- or any of the following OR'd together
    Audio1 = &H1
    Audio2 = &H2
    Audio3 = &H4
    Audio4 = &H8
    Audio5 = &H10
    Audio6 = &H20
    Audio7 = &H40
    Audio8 = &H80
    Audio9 = &H100
    Audio10 = &H200
    Audio11 = &H400
    Audio12 = &H800
    Audio13 = &H1000
    Audio14 = &H2000
    Audio15 = &H4000
    Audio16 = &H8000
    Audio17 = &H10000
    Audio18 = &H20000
    Audio19 = &H40000
    Audio20 = &H80000
    Audio21 = &H100000
    Audio22 = &H200000
    Audio23 = &H400000
    Audio24 = &H800000
    Video = &H2000000
End Enum

Public Enum ExtTransportEdit
    Invalid = ExtDeviceBase.Base + 652
    Executing = ExtDeviceBase.Base + 653
    Active = ExtDeviceBase.Base + 53
    Inactive = ExtDeviceBase.Base + 54
    Register = ExtDeviceBase.Base + 654
    Delete = ExtDeviceBase.Base + 655
    Hevent = ExtDeviceBase.Base + 656
    Test = ExtDeviceBase.Base + 657
    Immediate = ExtDeviceBase.Base + 658
    Mode = ExtDeviceBase.Base + 659
    ModeAssemble = ExtDeviceBase.Base + 660
    ModeInsert = ExtDeviceBase.Base + 661
    ModeCrashRecord = ExtDeviceBase.Base + 662
    ModeBookmarkTime = ExtDeviceBase.Base + 663
    ModeBookmarkChapter = ExtDeviceBase.Base + 664
    Master = ExtDeviceBase.Base + 666
    Track = ExtDeviceBase.Base + 667
    SourceInPoint = ExtDeviceBase.Base + 668
    SourceOutPoint = ExtDeviceBase.Base + 669
    RecInPoint = ExtDeviceBase.Base + 670
    RecOutPoint = ExtDeviceBase.Base + 671
    RehearseMode = ExtDeviceBase.Base + 672
    BVB = ExtDeviceBase.Base + 673
    VBV = ExtDeviceBase.Base + 674
    VVV = ExtDeviceBase.Base + 675
    Perform = ExtDeviceBase.Base + 676
    Abort = ExtDeviceBase.Base + 677
    TimeOut = ExtDeviceBase.Base + 678
    Seek = ExtDeviceBase.Base + 679
    SeekMode = ExtDeviceBase.Base + 680
    SeekEditIn = ExtDeviceBase.Base + 681
    SeekEditOut = ExtDeviceBase.Base + 682
    SeekPreRoll = ExtDeviceBase.Base + 683
    SeekPreRollCt = ExtDeviceBase.Base + 684
    SeekBookmark = ExtDeviceBase.Base + 685
    Offset = ExtDeviceBase.Base + 686
    PreRead = ExtDeviceBase.Base + 815
End Enum

<StructLayout(LayoutKind.Sequential)> _
Public Class VideoStreamConfigCaps
    Public guid As Guid
    Public VideoStandard As AnalogVideoStandard
    Public InputSize As Size
    Public MinCroppingSize As Size
    Public MaxCroppingSize As Size
    Public CropGranularityX As Integer
    Public CropGranularityY As Integer
    Public CropAlignX As Integer
    Public CropAlignY As Integer
    Public MinOutputSize As Size
    Public MaxOutputSize As Size
    Public OutputGranularityX As Integer
    Public OutputGranularityY As Integer
    Public StretchTapsX As Integer
    Public StretchTapsY As Integer
    Public ShrinkTapsX As Integer
    Public ShrinkTapsY As Integer
    Public MinFrameInterval As Long
    Public MaxFrameInterval As Long
    Public MinBitsPerSecond As Integer
    Public MaxBitsPerSecond As Integer
End Class

<StructLayout(LayoutKind.Sequential)> _
Public Class AudioStreamConfigCaps
    Public guid As Guid
    Public MinimumChannels As Integer
    Public MaximumChannels As Integer
    Public ChannelsGranularity As Integer
    Public MinimumBitsPerSample As Integer
    Public MaximumBitsPerSample As Integer
    Public BitsPerSampleGranularity As Integer
    Public MinimumSampleFrequency As Integer
    Public MaximumSampleFrequency As Integer
    Public SampleFrequencyGranularity As Integer
End Class

<StructLayout(LayoutKind.Sequential)> _
Public Structure Quality
    Public Type As QualityMessageType
    Public Proportion As Integer
    Public Late As Long
    Public TimeStamp As Long
End Structure

Public Enum QualityMessageType
    Famine
    Flood
End Enum

Public Enum CameraControlProperty
    Pan = 0
    Tilt
    Roll
    Zoom
    Exposure
    Iris
    Focus
End Enum

<Flags()> _
Public Enum CameraControlFlags
    None = &H0
    Auto = &H1
    Manual = &H2
End Enum


#If ALLOW_UNTESTED_INTERFACES Then

<ComImport(), System.Security.SuppressUnmanagedCodeSecurity(), Guid("D8D715A0-6E5E-11D0-B3F0-00AA003761C5"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)> _
Public Interface IAMVfwCaptureDialogs
    <PreserveSig()> _
    Function HasDialog(<[In]()> ByVal iDialog As VfwCaptureDialogs) As Integer

    ' HWND *
    <PreserveSig()> _
    Function ShowDialog(<[In]()> ByVal iDialog As VfwCaptureDialogs, <[In]()> ByVal hwnd As IntPtr) As Integer

    <PreserveSig()> _
    Function SendDriverMessage(<[In]()> ByVal iDialog As VfwCaptureDialogs, <[In]()> ByVal uMsg As Integer, <[In]()> ByVal dw1 As Integer, <[In]()> ByVal dw2 As Integer) As Integer
End Interface

' -------------------------------------------------------------------------------
' "This interface has been deprecated. Use IFilterMapper2::EnumMatchingFilters"
' -------------------------------------------------------------------------------
'<ComImport(), System.Security.SuppressUnmanagedCodeSecurity(), Guid("56a868a4-0ad4-11ce-b03a-0020af0ba770"), Obsolete("This interface has been deprecated.  Use IFilterMapper2::EnumMatchingFilters", False), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)> _
'Public Interface IEnumRegFilters
'    <PreserveSig()> _
'    Function [Next](<[In]()> ByVal cFilters As Integer, <Out(), MarshalAs(UnmanagedType.LPArray, SizeParamIndex:=0)> ByVal apRegFilter As RegFilter(), <[In]()> ByVal pcFetched As IntPtr) As Integer

'    <PreserveSig()> _
'    Function Skip(<[In]()> ByVal cFilters As Integer) As Integer

'    <PreserveSig()> _
'    Function Reset() As Integer

'    'Warning - IEnumRegFilters' is obsolete: 
'    'Use IFilterMapper2::EnumMatchingFilters'	
'    <PreserveSig()> _
'    Function Clone(<Out()> ByRef ppEnum As IEnumRegFilters) As Integer
'End Interface

' -------------------------------------------------------------------------------
' "This interface has been deprecated. Use IFilterMapper2::EnumMatchingFilters"
' -------------------------------------------------------------------------------
'<ComImport(), System.Security.SuppressUnmanagedCodeSecurity(), Guid("56a868a3-0ad4-11ce-b03a-0020af0ba770"), Obsolete("This interface has been deprecated.  Use IFilterMapper2.", False), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)> _
'Public Interface IFilterMapper
'    <PreserveSig()> _
'    Function RegisterFilter(<[In]()> ByVal clsid As Guid, <[In](), MarshalAs(UnmanagedType.LPWStr)> ByVal Name As String, <[In]()> ByVal dwMerit As Merit) As Integer

'    <PreserveSig()> _
'    Function RegisterFilterInstance(<[In]()> ByVal clsid As Guid, <[In](), MarshalAs(UnmanagedType.LPWStr)> ByVal Name As String, <Out()> ByRef MRId As Guid) As Integer

'    <PreserveSig()> _
'    Function RegisterPin(<[In]()> ByVal Filter As Guid, <[In](), MarshalAs(UnmanagedType.LPWStr)> ByVal Name As String, <[In](), MarshalAs(UnmanagedType.Bool)> ByVal bRendered As Boolean, <[In](), MarshalAs(UnmanagedType.Bool)> ByVal bOutput As Boolean, <[In](), MarshalAs(UnmanagedType.Bool)> ByVal bZero As Boolean, <[In](), MarshalAs(UnmanagedType.Bool)> ByVal bMany As Boolean, _
'     <[In]()> ByVal ConnectsToFilter As Guid, <[In](), MarshalAs(UnmanagedType.LPWStr)> ByVal ConnectsToPin As String) As Integer

'    <PreserveSig()> _
'    Function RegisterPinType(<[In]()> ByVal clsFilter As Guid, <[In](), MarshalAs(UnmanagedType.LPWStr)> ByVal strName As String, <[In]()> ByVal clsMajorType As Guid, <[In]()> ByVal clsSubType As Guid) As Integer

'    <PreserveSig()> _
'    Function UnregisterFilter(<[In]()> ByVal Filter As Guid) As Integer

'    <PreserveSig()> _
'    Function UnregisterFilterInstance(<[In]()> ByVal MRId As Guid) As Integer

'    <PreserveSig()> _
'    Function UnregisterPin(<[In]()> ByVal Filter As Guid, <[In](), MarshalAs(UnmanagedType.LPWStr)> ByVal Name As String) As Integer

'    <PreserveSig()> _
'    Function EnumMatchingFilters(<Out()> ByRef ppEnum As IEnumRegFilters, <[In]()> ByVal dwMerit As Merit, <[In](), MarshalAs(UnmanagedType.Bool)> ByVal bInputNeeded As Boolean, <[In]()> ByVal clsInMaj As Guid, <[In]()> ByVal clsInSub As Guid, <[In](), MarshalAs(UnmanagedType.Bool)> ByVal bRender As Boolean, _
'     <[In](), MarshalAs(UnmanagedType.Bool)> ByVal bOututNeeded As Boolean, <[In]()> ByVal clsOutMaj As Guid, <[In]()> ByVal clsOutSub As Guid) As Integer
'End Interface

<ComImport(), System.Security.SuppressUnmanagedCodeSecurity(), Guid("56a868a0-0ad4-11ce-b03a-0020af0ba770"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)> _
Public Interface IOverlayNotify

    ' PALETTEENTRY *
    <PreserveSig()> _
    Function OnPaletteChange(<[In]()> ByVal dwColors As Integer, <[In]()> ByVal pPalette As IntPtr) As Integer

    <PreserveSig()> _
    Function OnClipChange(<[In]()> ByVal pSourceRect As Rectangle, <[In]()> ByVal pDestinationRect As Rectangle, <[In]()> ByVal pRgnData As RgnData) As Integer

    <PreserveSig()> _
    Function OnColorKeyChange(<[In]()> ByVal pColorKey As ColorKey) As Integer

    <PreserveSig()> _
    Function OnPositionChange(<[In]()> ByVal pSourceRect As Rectangle, <[In]()> ByVal pDestinationRect As Rectangle) As Integer
End Interface

<ComImport(), System.Security.SuppressUnmanagedCodeSecurity(), Guid("680EFA10-D535-11D1-87C8-00A0C9223196"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)> _
Public Interface IOverlayNotify2
    Inherits IOverlayNotify

    ' PALETTEENTRY *
    <PreserveSig()> _
    Shadows Function OnPaletteChange(<[In]()> ByVal dwColors As Integer, <[In]()> ByVal pPalette As IntPtr) As Integer

    <PreserveSig()> _
    Shadows Function OnClipChange(<[In]()> ByVal pSourceRect As Rectangle, <[In]()> ByVal pDestinationRect As Rectangle, <[In]()> ByVal pRgnData As RgnData) As Integer

    <PreserveSig()> _
    Shadows Function OnColorKeyChange(<[In]()> ByVal pColorKey As ColorKey) As Integer

    <PreserveSig()> _
    Shadows Function OnPositionChange(<[In]()> ByVal pSourceRect As Rectangle, <[In]()> ByVal pDestinationRect As Rectangle) As Integer

    <PreserveSig()> _
    Function OnDisplayChange(ByVal hMonitor As IntPtr) As Integer ' HMONITOR
End Interface

<ComImport(), System.Security.SuppressUnmanagedCodeSecurity(), Guid("56a868a1-0ad4-11ce-b03a-0020af0ba770"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)> _
Public Interface IOverlay

    ' PALETTEENTRY **
    <PreserveSig()> _
    Function GetPalette(<Out()> ByRef pdwColors As Integer, <Out()> ByRef ppPalette As IntPtr) As Integer

    ' PALETTEENTRY *
    <PreserveSig()> _
    Function SetPalette(<[In]()> ByVal dwColors As Integer, <[In]()> ByVal pPalette As IntPtr) As Integer

    <PreserveSig()> _
    Function GetDefaultColorKey(<Out()> ByRef pColorKey As ColorKey) As Integer

    <PreserveSig()> _
    Function GetColorKey(<Out()> ByVal pColorKey As ColorKey) As Integer

    <PreserveSig()> _
    Function SetColorKey(<[In]()> ByRef pColorKey As ColorKey) As Integer

    <PreserveSig()> _
    Function GetWindowHandle(<Out()> ByRef pHwnd As IntPtr) As Integer

    ' HWND *
    <PreserveSig()> _
    Function GetClipList(<Out()> ByRef pSourceRect As Rectangle, <Out()> ByRef pDestinationRect As Rectangle, <Out()> ByRef ppRgnData As RgnData) As Integer

    <PreserveSig()> _
    Function GetVideoPosition(<Out()> ByRef pSourceRect As Rectangle, <Out()> ByRef pDestinationRect As Rectangle) As Integer

    <PreserveSig()> _
    Function Advise(<[In]()> ByVal pOverlayNotify As IOverlayNotify, <[In]()> ByVal dwInterests As Advise) As Integer

    <PreserveSig()> _
    Function Unadvise() As Integer
End Interface

<ComImport(), System.Security.SuppressUnmanagedCodeSecurity(), Guid("bf87b6e0-8c27-11d0-b3f0-00aa003761c5"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown), Obsolete("The ICaptureGraphBuilder interface is deprecated. Use ICaptureGraphBuilder2 instead.", False)> _
Public Interface ICaptureGraphBuilder
    <PreserveSig()> _
    Function SetFiltergraph(<[In]()> ByVal pfg As IGraphBuilder) As Integer

    <PreserveSig()> _
    Function GetFiltergraph(<Out()> ByRef ppfg As IGraphBuilder) As Integer

    <PreserveSig()> _
    Function SetOutputFileName(<[In](), MarshalAs(UnmanagedType.LPStruct)> ByVal pType As Guid, <[In](), MarshalAs(UnmanagedType.LPWStr)> ByVal lpstrFile As String, <Out()> ByRef ppbf As IBaseFilter, <Out()> ByRef ppSink As IFileSinkFilter) As Integer

    <PreserveSig()> _
    Function FindInterface(<[In](), MarshalAs(UnmanagedType.LPStruct)> ByVal pCategory As Guid, <[In]()> ByVal pf As IBaseFilter, <[In](), MarshalAs(UnmanagedType.LPStruct)> ByVal riid As Guid, <Out(), MarshalAs(UnmanagedType.IUnknown)> ByRef ppint As Object) As Integer

    <PreserveSig()> _
    Function RenderStream(<[In](), MarshalAs(UnmanagedType.LPStruct)> ByVal pCategory As Guid, <[In](), MarshalAs(UnmanagedType.IUnknown)> ByVal pSource As Object, <[In]()> ByVal pfCompressor As IBaseFilter, <[In]()> ByVal pfRenderer As IBaseFilter) As Integer

    <PreserveSig()> _
    Function ControlStream(<[In](), MarshalAs(UnmanagedType.LPStruct)> ByVal pCategory As Guid, <[In]()> ByVal pFilter As IBaseFilter, <[In]()> ByVal pstart As Long, <[In]()> ByVal pstop As Long, <[In]()> ByVal wStartCookie As Short, <[In]()> ByVal wStopCookie As Short) As Integer

    <PreserveSig()> _
    Function AllocCapFile(<[In](), MarshalAs(UnmanagedType.LPWStr)> ByVal lpstrFile As String, <[In]()> ByVal dwlSize As Long) As Integer

    <PreserveSig()> _
    Function CopyCaptureFile(<[In](), MarshalAs(UnmanagedType.LPWStr)> ByVal lpwstrOld As String, <[In](), MarshalAs(UnmanagedType.LPWStr)> ByVal lpwstrNew As String, <[In]()> ByVal fAllowEscAbort As Integer, <[In]()> ByVal pFilter As IAMCopyCaptureFileProgress) As Integer
End Interface

<ComImport(), System.Security.SuppressUnmanagedCodeSecurity(), Guid("56a868bf-0ad4-11ce-b03a-0020af0ba770"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)> _
Public Interface IStreamBuilder
    <PreserveSig()> _
    Function Render(<[In]()> ByVal ppinOut As IPin, <[In]()> ByVal pGraph As IGraphBuilder) As Integer

    <PreserveSig()> _
    Function Backout(<[In]()> ByVal ppinOut As IPin, <[In]()> ByVal pGraph As IGraphBuilder) As Integer
End Interface

<ComImport(), System.Security.SuppressUnmanagedCodeSecurity(), Guid("56a868ad-0ad4-11ce-b03a-0020af0ba770"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)> _
Public Interface IResourceConsumer
    <PreserveSig()> _
    Function AcquireResource(<[In]()> ByVal idResource As Integer) As Integer

    <PreserveSig()> _
    Function ReleaseResource(<[In]()> ByVal idResource As Integer) As Integer
End Interface

<ComImport(), System.Security.SuppressUnmanagedCodeSecurity(), Guid("56a868ac-0ad4-11ce-b03a-0020af0ba770"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)> _
Public Interface IResourceManager
    <PreserveSig()> _
    Function Register(<[In](), MarshalAs(UnmanagedType.LPWStr)> ByVal pName As String, <[In]()> ByVal cResource As Integer, <Out()> ByRef plToken As Integer) As Integer

    ' int *
    <PreserveSig()> _
    Function RegisterGroup(<[In](), MarshalAs(UnmanagedType.LPWStr)> ByVal pName As String, <[In]()> ByVal cResource As Integer, <[In]()> ByVal palTokens As IntPtr, <Out()> ByRef plToken As Integer) As Integer

    <PreserveSig()> _
    Function RequestResource(<[In]()> ByVal idResource As Integer, <[In](), MarshalAs(UnmanagedType.IUnknown)> ByVal pFocusObject As Object, <[In]()> ByVal pConsumer As IResourceConsumer) As Integer

    <PreserveSig()> _
    Function NotifyAcquire(<[In]()> ByVal idResource As Integer, <[In]()> ByVal pConsumer As IResourceConsumer, <[In]()> ByVal hr As Integer) As Integer

    <PreserveSig()> _
    Function NotifyRelease(<[In]()> ByVal idResource As Integer, <[In]()> ByVal pConsumer As IResourceConsumer, <[In](), MarshalAs(UnmanagedType.Bool)> ByVal bStillWant As Boolean) As Integer

    <PreserveSig()> _
    Function CancelRequest(<[In]()> ByVal idResource As Integer, <[In]()> ByVal pConsumer As IResourceConsumer) As Integer

    <PreserveSig()> _
    Function SetFocus(<[In](), MarshalAs(UnmanagedType.IUnknown)> ByVal pFocusObject As Object) As Integer

    <PreserveSig()> _
    Function ReleaseFocus(<[In](), MarshalAs(UnmanagedType.IUnknown)> ByVal pFocusObject As Object) As Integer
End Interface

<ComImport(), System.Security.SuppressUnmanagedCodeSecurity(), Guid("56a868af-0ad4-11ce-b03a-0020af0ba770"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)> _
Public Interface IDistributorNotify
    <PreserveSig()> _
    Function [Stop]() As Integer

    <PreserveSig()> _
    Function Pause() As Integer

    <PreserveSig()> _
    Function Run(ByVal tStart As Long) As Integer

    <PreserveSig()> _
    Function SetSyncSource(<[In]()> ByVal pClock As IReferenceClock) As Integer

    <PreserveSig()> _
    Function NotifyGraphChange() As Integer
End Interface

<ComImport(), System.Security.SuppressUnmanagedCodeSecurity(), Guid("211A8765-03AC-11d1-8D13-00AA00BD8339"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)> _
Public Interface IBPCSatelliteTuner
    Inherits IAMTuner

    <PreserveSig()> _
    Shadows Function put_Channel(<[In]()> ByVal lChannel As Integer, <[In]()> ByVal lVideoSubChannel As AMTunerSubChannel, <[In]()> ByVal lAudioSubChannel As AMTunerSubChannel) As Integer

    <PreserveSig()> _
    Shadows Function get_Channel(<Out()> ByRef plChannel As Integer, <Out()> ByRef plVideoSubChannel As AMTunerSubChannel, <Out()> ByRef plAudioSubChannel As AMTunerSubChannel) As Integer

    <PreserveSig()> _
    Shadows Function ChannelMinMax(<Out()> ByRef lChannelMin As Integer, <Out()> ByRef lChannelMax As Integer) As Integer

    <PreserveSig()> _
    Shadows Function put_CountryCode(<[In]()> ByVal lCountryCode As Integer) As Integer

    <PreserveSig()> _
    Shadows Function get_CountryCode(<Out()> ByRef plCountryCode As Integer) As Integer

    <PreserveSig()> _
    Shadows Function put_TuningSpace(<[In]()> ByVal lTuningSpace As Integer) As Integer

    <PreserveSig()> _
    Shadows Function get_TuningSpace(<Out()> ByRef plTuningSpace As Integer) As Integer

    <PreserveSig()> _
    Shadows Function Logon(<[In]()> ByVal hCurrentUser As IntPtr) As Integer

    <PreserveSig()> _
    Shadows Function Logout() As Integer

    <PreserveSig()> _
    Shadows Function SignalPresent(<Out()> ByRef plSignalStrength As AMTunerSignalStrength) As Integer

    <PreserveSig()> _
    Shadows Function put_Mode(<[In]()> ByVal lMode As AMTunerModeType) As Integer

    <PreserveSig()> _
    Shadows Function get_Mode(<Out()> ByRef plMode As AMTunerModeType) As Integer

    <PreserveSig()> _
    Shadows Function GetAvailableModes(<Out()> ByRef plModes As AMTunerModeType) As Integer

    <PreserveSig()> _
    Shadows Function RegisterNotificationCallBack(<[In]()> ByVal pNotify As IAMTunerNotification, <[In]()> ByVal lEvents As AMTunerEventType) As Integer

    <PreserveSig()> _
    Shadows Function UnRegisterNotificationCallBack(<[In]()> ByVal pNotify As IAMTunerNotification) As Integer

    <PreserveSig()> _
    Function get_DefaultSubChannelTypes(<Out()> ByRef plDefaultVideoType As Integer, <Out()> ByRef plDefaultAudioType As Integer) As Integer

    <PreserveSig()> _
    Function put_DefaultSubChannelTypes(<[In]()> ByVal lDefaultVideoType As Integer, <[In]()> ByVal lDefaultAudioType As Integer) As Integer

    <PreserveSig()> _
    Function IsTapingPermitted() As Integer
End Interface

<ComImport(), System.Security.SuppressUnmanagedCodeSecurity(), Guid("83EC1C33-23D1-11d1-99E6-00A0C9560266"), Obsolete("This interface has been deprecated.", False), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)> _
Public Interface IAMTVAudioNotification
    <PreserveSig()> _
    Function OnEvent(<[In]()> ByVal [Event] As AMTVAudioEventType) As Integer
End Interface

<ComImport(), System.Security.SuppressUnmanagedCodeSecurity(), Guid("C6E133B0-30AC-11d0-A18C-00A0C9118956"), Obsolete("This interface has been deprecated.", False), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)> _
Public Interface IAMAnalogVideoEncoder
    <PreserveSig()> _
    Function get_AvailableTVFormats(<Out()> ByRef lAnalogVideoStandard As AnalogVideoStandard) As Integer

    <PreserveSig()> _
    Function put_TVFormat(<[In]()> ByVal lAnalogVideoStandard As AnalogVideoStandard) As Integer

    <PreserveSig()> _
    Function get_TVFormat(<Out()> ByRef plAnalogVideoStandard As AnalogVideoStandard) As Integer

    <PreserveSig()> _
    Function put_CopyProtection(<[In]()> ByVal lVideoCopyProtection As VideoCopyProtectionType) As Integer

    <PreserveSig()> _
    Function get_CopyProtection(<Out()> ByRef lVideoCopyProtection As VideoCopyProtectionType) As Integer

    <PreserveSig()> _
    Function put_CCEnable(<[In]()> ByVal lCCEnable As Integer) As Integer

    <PreserveSig()> _
    Function get_CCEnable(<Out()> ByRef lCCEnable As Integer) As Integer
End Interface

<ComImport(), System.Security.SuppressUnmanagedCodeSecurity(), Guid("F938C991-3029-11cf-8C44-00AA006B6814"), Obsolete("This interface has been deprecated.", False), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)> _
Public Interface IAMPhysicalPinInfo
    <PreserveSig()> _
    Function GetPhysicalType(<Out()> ByRef pType As PhysicalConnectorType, <Out(), MarshalAs(UnmanagedType.LPWStr)> ByRef ppszType As String) As Integer
End Interface

<ComImport(), System.Security.SuppressUnmanagedCodeSecurity(), Guid("9B496CE1-811B-11cf-8C77-00AA006B6814"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)> _
Public Interface IAMTimecodeReader
    <PreserveSig()> _
    Function GetTCRMode(<[In]()> ByVal Param As Integer, <Out()> ByRef pValue As Integer) As Integer

    <PreserveSig()> _
    Function SetTCRMode(<[In]()> ByVal Param As Integer, <[In]()> ByVal Value As Integer) As Integer

    <PreserveSig()> _
    Function put_VITCLine(<[In]()> ByVal Line As Integer) As Integer

    <PreserveSig()> _
    Function get_VITCLine(<Out()> ByRef pLine As Integer) As Integer

    <PreserveSig()> _
    Function GetTimecode(<Out()> ByRef pTimecodeSample As TimeCodeSample) As Integer
End Interface

<ComImport(), System.Security.SuppressUnmanagedCodeSecurity(), Guid("9B496CE0-811B-11cf-8C77-00AA006B6814"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)> _
Public Interface IAMTimecodeGenerator
    <PreserveSig()> _
    Function GetTCGMode(<[In]()> ByVal Param As Integer, <Out()> ByRef pValue As Integer) As Integer

    <PreserveSig()> _
    Function SetTCGMode(<[In]()> ByVal Param As Integer, <[In]()> ByVal Value As Integer) As Integer

    <PreserveSig()> _
    Function put_VITCLine(<[In]()> ByVal Line As Integer) As Integer

    <PreserveSig()> _
    Function get_VITCLine(<Out()> ByRef pLine As Integer) As Integer

    <PreserveSig()> _
    Function SetTimecode(<[In]()> ByVal pTimecodeSample As TimeCodeSample) As Integer

    <PreserveSig()> _
    Function GetTimecode(<Out()> ByVal pTimecodeSample As TimeCodeSample) As Integer
End Interface

<ComImport(), System.Security.SuppressUnmanagedCodeSecurity(), Guid("9B496CE2-811B-11cf-8C77-00AA006B6814"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)> _
Public Interface IAMTimecodeDisplay
    <PreserveSig()> _
    Function GetTCDisplayEnable(<Out()> ByRef pState As Integer) As Integer

    <PreserveSig()> _
    Function SetTCDisplayEnable(<[In]()> ByVal State As Integer) As Integer

    <PreserveSig()> _
    Function GetTCDisplay(<[In]()> ByVal Param As Integer, <Out()> ByRef pValue As Integer) As Integer

    <PreserveSig()> _
    Function SetTCDisplay(<[In]()> ByVal Param As Integer, <[In]()> ByVal Value As Integer) As Integer
End Interface

<ComImport(), System.Security.SuppressUnmanagedCodeSecurity(), Guid("c6545bf0-e76b-11d0-bd52-00a0c911ce86"), Obsolete("This interface has been deprecated.", False), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)> _
Public Interface IAMDevMemoryAllocator
    <PreserveSig()> _
    Function GetInfo(<Out()> ByRef pdwcbTotalFree As Integer, <Out()> ByRef pdwcbLargestFree As Integer, <Out()> ByRef pdwcbTotalMemory As Integer, <Out()> ByRef pdwcbMinimumChunk As Integer) As Integer

    ' BYTE *
    <PreserveSig()> _
    Function CheckMemory(<[In]()> ByVal pBuffer As IntPtr) As Integer

    ' BYTE **
    <PreserveSig()> _
    Function Alloc(<Out()> ByRef ppBuffer As IntPtr, <[In](), Out()> ByRef pdwcbBuffer As Integer) As Integer

    ' BYTE *
    <PreserveSig()> _
    Function Free(<[In]()> ByVal pBuffer As IntPtr) As Integer

    <PreserveSig()> _
    Function GetDevMemoryObject(<Out(), MarshalAs(UnmanagedType.IUnknown)> ByRef ppUnkInnner As Object, <[In](), MarshalAs(UnmanagedType.IUnknown)> ByVal pUnkOuter As Object) As Integer
End Interface

<ComImport(), System.Security.SuppressUnmanagedCodeSecurity(), Guid("c6545bf1-e76b-11d0-bd52-00a0c911ce86"), Obsolete("This interface has been deprecated.", False), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)> _
Public Interface IAMDevMemoryControl
    <PreserveSig()> _
    Function QueryWriteSync() As Integer

    <PreserveSig()> _
    Function WriteSync() As Integer

    <PreserveSig()> _
    Function GetDevId(<Out()> ByRef pdwDevId As Integer) As Integer
End Interface

<ComImport(), System.Security.SuppressUnmanagedCodeSecurity(), Guid("48efb120-ab49-11d2-aed2-00a0c995e8d5"), Obsolete("This interface has been deprecated.", False), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)> _
Public Interface IDrawVideoImage
    <PreserveSig()> _
    Function DrawVideoImageBegin() As Integer

    <PreserveSig()> _
    Function DrawVideoImageEnd() As Integer

    ' HDC
    <PreserveSig()> _
    Function DrawVideoImageDraw(<[In]()> ByVal hdc As IntPtr, <[In]()> ByVal lprcSrc As Rectangle, <[In]()> ByVal lprcDst As Rectangle) As Integer
End Interface

<ComImport(), System.Security.SuppressUnmanagedCodeSecurity(), Guid("2e5ea3e0-e924-11d2-b6da-00a0c995e8df"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)> _
Public Interface IDecimateVideoImage
    <PreserveSig()> _
    Function SetDecimationImageSize(<[In]()> ByVal lWidth As Integer, <[In]()> ByVal lHeight As Integer) As Integer

    <PreserveSig()> _
    Function ResetDecimationImageSize() As Integer
End Interface

<ComImport(), System.Security.SuppressUnmanagedCodeSecurity(), Guid("F185FE76-E64E-11d2-B76E-00C04FB6BD3D"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)> _
Public Interface IAMPushSource
    <PreserveSig()> _
    Function GetPushSourceFlags(<Out()> ByRef pFlags As AMPushSourceFlags) As Integer

    <PreserveSig()> _
    Function SetPushSourceFlags(<[In]()> ByVal Flags As AMPushSourceFlags) As Integer

    <PreserveSig()> _
    Function SetStreamOffset(<[In]()> ByVal rtOffset As Long) As Integer

    <PreserveSig()> _
    Function GetStreamOffset(<Out()> ByRef prtOffset As Long) As Integer

    <PreserveSig()> _
    Function GetMaxStreamOffset(<Out()> ByRef prtMaxOffset As Long) As Integer

    <PreserveSig()> _
    Function SetMaxStreamOffset(<[In]()> ByVal rtMaxOffset As Long) As Integer
End Interface

<ComImport(), System.Security.SuppressUnmanagedCodeSecurity(), Guid("f90a6130-b658-11d2-ae49-0000f8754b99"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)> _
Public Interface IAMDeviceRemoval
    <PreserveSig()> _
    Function DeviceInfo(<Out()> ByRef pclsidInterfaceClass As Guid, <Out(), MarshalAs(UnmanagedType.LPWStr)> ByRef pwszSymbolicLink As String) As Integer

    <PreserveSig()> _
    Function Reassociate() As Integer

    <PreserveSig()> _
    Function Disassociate() As Integer
End Interface

<ComImport(), System.Security.SuppressUnmanagedCodeSecurity(), Guid("62EA93BA-EC62-11d2-B770-00C04FB6BD3D"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)> _
Public Interface IAMLatency
    <PreserveSig()> _
    Function GetLatency(ByRef prtLatency As Long) As Integer
End Interface

<ComImport(), System.Security.SuppressUnmanagedCodeSecurity(), Guid("945C1566-6202-46fc-96C7-D87F289C6534"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)> _
Public Interface IEnumStreamIdMap

    ' STREAM_ID_MAP *
    <PreserveSig()> _
    Function [Next](<[In]()> ByVal cRequest As Integer, <Out(), MarshalAs(UnmanagedType.LPArray, SizeParamIndex:=0)> ByVal pStreamIdMap As StreamIdMap(), <[In]()> ByVal pcReceived As IntPtr) As Integer

    <PreserveSig()> _
    Function Skip(<[In]()> ByVal cRecords As Integer) As Integer

    <PreserveSig()> _
    Function Reset() As Integer

    <PreserveSig()> _
    Function Clone(<Out()> ByRef ppIEnumStreamIdMap As IEnumStreamIdMap) As Integer
End Interface

<ComImport(), System.Security.SuppressUnmanagedCodeSecurity(), Guid("901db4c7-31ce-41a2-85dc-8fa0bf41b8da"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)> _
Public Interface ICodecAPI
    <PreserveSig()> _
    Function IsSupported(<[In](), MarshalAs(UnmanagedType.LPStruct)> ByVal Api As Guid) As Integer

    <PreserveSig()> _
    Function IsModifiable(<[In](), MarshalAs(UnmanagedType.LPStruct)> ByVal Api As Guid) As Integer

    <PreserveSig()> _
    Function GetParameterRange(<[In](), MarshalAs(UnmanagedType.LPStruct)> ByVal Api As Guid, <Out()> ByRef ValueMin As Object, <Out()> ByRef ValueMax As Object, <Out()> ByRef SteppingDelta As Object) As Integer

    <PreserveSig()> _
    Function GetParameterValues(<[In](), MarshalAs(UnmanagedType.LPStruct)> ByVal Api As Guid, ByRef ip As IntPtr, <Out()> ByRef ValuesCount As Integer) As Integer

    <PreserveSig()> _
    Function GetDefaultValue(<[In](), MarshalAs(UnmanagedType.LPStruct)> ByVal Api As Guid, <Out(), MarshalAs(UnmanagedType.Struct)> ByRef Value As Object) As Integer

    <PreserveSig()> _
    Function GetValue(<[In](), MarshalAs(UnmanagedType.LPStruct)> ByVal Api As Guid, <Out(), MarshalAs(UnmanagedType.Struct)> ByRef Value As Object) As Integer

    <PreserveSig()> _
    Function SetValue(<[In](), MarshalAs(UnmanagedType.LPStruct)> ByVal Api As Guid, <[In]()> ByRef Value As Object) As Integer

    <PreserveSig()> _
    Function RegisterForEvent(<[In](), MarshalAs(UnmanagedType.LPStruct)> ByVal Api As Guid, <[In]()> ByVal userData As IntPtr) As Integer

    <PreserveSig()> _
    Function UnregisterForEvent(<[In](), MarshalAs(UnmanagedType.LPStruct)> ByVal Api As Guid) As Integer

    <PreserveSig()> _
    Function SetAllDefaults() As Integer

    <PreserveSig()> _
    Function SetValueWithNotify(<[In](), MarshalAs(UnmanagedType.LPStruct)> ByVal Api As Guid, <[In]()> ByVal Value As Object, <Out()> ByRef ChangedParam As Guid(), <Out()> ByRef ChangedParamCount As Integer) As Integer

    <PreserveSig()> _
    Function SetAllDefaultsWithNotify(<Out()> ByRef ChangedParam As Guid(), <Out()> ByRef ChangedParamCount As Integer) As Integer

    <PreserveSig()> _
    Function GetAllSettings(<[In]()> ByVal pStream As IStream) As Integer

    <PreserveSig()> _
    Function SetAllSettings(<[In]()> ByVal pStream As IStream) As Integer


    <PreserveSig()> _
    Function SetAllSettingsWithNotify(<[In]()> ByVal pStream As IStream, <Out()> ByRef ChangedParam As Guid(), <Out()> ByRef ChangedParamCount As Integer) As Integer
End Interface

<ComImport(), System.Security.SuppressUnmanagedCodeSecurity(), Guid("70423839-6ACC-4b23-B079-21DBF08156A5"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)> _
<Obsolete("This interface is deprecated and is maintained for backward compatibility only. New applications and drivers should use the ICodecAPI interface.")> _
Public Interface IEncoderAPI
    <PreserveSig()> _
    Function IsSupported(<[In](), MarshalAs(UnmanagedType.LPStruct)> ByVal Api As Guid) As Integer

    <PreserveSig()> _
    Function IsAvailable(<[In](), MarshalAs(UnmanagedType.LPStruct)> ByVal Api As Guid) As Integer

    <PreserveSig()> _
    Function GetParameterRange(<[In](), MarshalAs(UnmanagedType.LPStruct)> ByVal Api As Guid, <Out()> ByRef ValueMin As Object, <Out()> ByRef ValueMax As Object, <Out()> ByRef SteppingDelta As Object) As Integer

    <PreserveSig()> _
    Function GetParameterValues(<[In](), MarshalAs(UnmanagedType.LPStruct)> ByVal Api As Guid, <Out()> ByRef Values As Object(), <Out()> ByRef ValuesCount As Integer) As Integer

    <PreserveSig()> _
    Function GetDefaultValue(<[In](), MarshalAs(UnmanagedType.LPStruct)> ByVal Api As Guid, <Out()> ByRef Value As Object) As Integer

    <PreserveSig()> _
    Function GetValue(<[In](), MarshalAs(UnmanagedType.LPStruct)> ByVal Api As Guid, <Out()> ByRef Value As Object) As Integer

    <PreserveSig()> _
    Function SetValue(<[In](), MarshalAs(UnmanagedType.LPStruct)> ByVal Api As Guid, <[In]()> ByVal Value As Object) As Integer
End Interface

<ComImport(), System.Security.SuppressUnmanagedCodeSecurity(), Guid("02997C3B-8E1B-460e-9270-545E0DE9563E"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)> _
Public Interface IVideoEncoder
    Inherits ICodecAPI ' IEncoderAPI

    <PreserveSig()> _
    Shadows Function IsSupported(<[In]()> ByVal Api As Guid) As Integer

    <PreserveSig()> _
    Shadows Function IsAvailable(<[In]()> ByVal Api As Guid) As Integer

    <PreserveSig()> _
    Shadows Function GetParameterRange(<[In]()> ByVal Api As Guid, <Out()> ByRef ValueMin As Object, <Out()> ByRef ValueMax As Object, <Out()> ByRef SteppingDelta As Object) As Integer

    <PreserveSig()> _
    Shadows Function GetParameterValues(<[In]()> ByVal Api As Guid, <Out()> ByRef Values As Object(), <Out()> ByRef ValuesCount As Integer) As Integer

    <PreserveSig()> _
    Shadows Function GetDefaultValue(<[In]()> ByVal Api As Guid, <Out()> ByRef Value As Object) As Integer

    <PreserveSig()> _
    Shadows Function GetValue(<[In]()> ByVal Api As Guid, <Out()> ByRef Value As Object) As Integer

    <PreserveSig()> _
    Shadows Function SetValue(<[In]()> ByVal Api As Guid, <[In]()> ByVal Value As Object) As Integer

End Interface

<ComImport(), System.Security.SuppressUnmanagedCodeSecurity(), Guid("6feded3e-0ff1-4901-a2f1-43f7012c8515"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)> _
Public Interface IAMCertifiedOutputProtection

    ' BYTE **
    <PreserveSig()> _
    Function KeyExchange(<Out()> ByRef pRandom As Guid, <Out()> ByRef VarLenCertGH As IntPtr, <Out()> ByRef pdwLengthCertGH As Integer) As Integer

    <PreserveSig()> _
    Function SessionSequenceStart(<[In](), MarshalAs(UnmanagedType.LPArray)> ByVal pSig As Byte()) As Integer

    <PreserveSig()> _
    Function ProtectionCommand(<[In](), MarshalAs(UnmanagedType.LPStruct)> ByVal cmd As AMCOPPCommand) As Integer

    <PreserveSig()> _
    Function ProtectionStatus(<[In]()> ByVal pStatusInput As AMCOPPStatusInput, <Out()> ByRef pStatusOutput As AMCOPPStatusOutput) As Integer
End Interface

#End If

<ComImport(), System.Security.SuppressUnmanagedCodeSecurity(), Guid("56a868ab-0ad4-11ce-b03a-0020af0ba770"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)> _
Public Interface IGraphVersion
    <PreserveSig()> _
    Function QueryVersion(<Out()> ByRef pVersion As Integer) As Integer
End Interface

<ComImport(), System.Security.SuppressUnmanagedCodeSecurity(), Guid("b8e8bd60-0bfe-11d0-af91-00aa00b67a42"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)> _
Public Interface IIPDVDec
    <PreserveSig()> _
    Function get_IPDisplay(<Out()> ByRef displayPix As DVDecoderResolution) As Integer

    <PreserveSig()> _
    Function put_IPDisplay(<[In]()> ByVal displayPix As DVDecoderResolution) As Integer
End Interface

<ComImport(), System.Security.SuppressUnmanagedCodeSecurity(), Guid("92a3a302-da7c-4a1f-ba7e-1802bb5d2d02"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)> _
Public Interface IDVSplitter
    <PreserveSig()> _
    Function DiscardAlternateVideoFrames(<[In](), MarshalAs(UnmanagedType.Bool)> ByVal nDiscard As Boolean) As Integer
End Interface

<ComImport(), System.Security.SuppressUnmanagedCodeSecurity(), Guid("58473A19-2BC8-4663-8012-25F81BABDDD1"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)> _
Public Interface IDVRGB219
    <PreserveSig()> _
    Function SetRGB219(<[In](), MarshalAs(UnmanagedType.Bool)> ByVal bState As Boolean) As Integer
End Interface

<ComImport(), System.Security.SuppressUnmanagedCodeSecurity(), Guid("36b73881-c2c8-11cf-8b46-00805f6cef60"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)> _
Public Interface IAMStreamControl
    <PreserveSig()> _
    Function StartAt(<[In]()> ByVal ptStart As DsLong, <[In]()> ByVal dwCookie As Integer) As Integer

    <PreserveSig()> _
    Function StopAt(<[In]()> ByVal ptStop As DsLong, <[In](), MarshalAs(UnmanagedType.Bool)> ByVal bSendExtra As Boolean, <[In]()> ByVal dwCookie As Integer) As Integer

    <PreserveSig()> _
    Function GetInfo(<Out()> ByRef pInfo As AMStreamInfo) As Integer
End Interface

<ComImport(), System.Security.SuppressUnmanagedCodeSecurity(), Guid("8E1C39A1-DE53-11cf-AA63-0080C744528D"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)> _
Public Interface IAMOpenProgress
    <PreserveSig()> _
    Function QueryProgress(<Out()> ByRef pllTotal As Long, <Out()> ByRef pllCurrent As Long) As Integer

    <PreserveSig()> _
    Function AbortOperation() As Integer
End Interface

<ComImport(), System.Security.SuppressUnmanagedCodeSecurity(), Guid("2dd74950-a890-11d1-abe8-00a0c905f375"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)> _
Public Interface IAMFilterMiscFlags
    <PreserveSig()> _
    Function GetMiscFlags() As Integer
End Interface

<ComImport(), System.Security.SuppressUnmanagedCodeSecurity(), Guid("5738E040-B67F-11d0-BD4D-00A0C911CE86"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)> _
Public Interface IPersistMediaPropertyBag
    Inherits IPersist

    <PreserveSig()> _
    Shadows Function GetClassID(<Out()> ByRef pClassID As Guid) As Integer

    <PreserveSig()> _
    Function InitNew() As Integer

    <PreserveSig()> _
    Function Load(<[In]()> ByVal pPropBag As IMediaPropertyBag, <[In]()> ByVal pErrorLog As IErrorLog) As Integer

    <PreserveSig()> _
    Function Save(ByVal pPropBag As IMediaPropertyBag, <[In](), MarshalAs(UnmanagedType.Bool)> ByVal fClearDirty As Boolean, <[In](), MarshalAs(UnmanagedType.Bool)> ByVal fSaveAllProperties As Boolean) As Integer
End Interface

<ComImport(), System.Security.SuppressUnmanagedCodeSecurity(), Guid("6025A880-C0D5-11d0-BD4E-00A0C911CE86"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)> _
Public Interface IMediaPropertyBag
    Inherits IPropertyBag

    <PreserveSig()> _
    Shadows Function Read(<[In](), MarshalAs(UnmanagedType.LPWStr)> ByVal pszPropName As String, <Out(), MarshalAs(UnmanagedType.Struct)> ByRef pVar As Object, <[In]()> ByVal pErrorLog As IErrorLog) As Integer

    <PreserveSig()> _
    Shadows Function Write(<[In](), MarshalAs(UnmanagedType.LPWStr)> ByVal pszPropName As String, <[In]()> ByRef pVar As Object) As Integer

    <PreserveSig()> _
    Function EnumProperty(<[In]()> ByVal iProperty As Integer, <Out()> ByRef pvarPropertyName As Object, <Out()> ByRef pvarPropertyValue As Object) As Integer
End Interface

<ComImport(), System.Security.SuppressUnmanagedCodeSecurity(), Guid("632105FA-072E-11d3-8AF9-00C04FB6BD3D"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)> _
Public Interface IAMGraphStreams
    <PreserveSig()> _
    Function FindUpstreamInterface(<[In]()> ByVal pPin As IPin, <[In](), MarshalAs(UnmanagedType.LPStruct)> ByVal riid As Guid, <Out(), MarshalAs(UnmanagedType.IUnknown)> ByRef ppvInterface As Object, <[In]()> ByVal dwFlags As AMIntfSearchFlags) As Integer

    <PreserveSig()> _
    Function SyncUsingStreamOffset(<[In](), MarshalAs(UnmanagedType.Bool)> ByVal bUseStreamOffset As Boolean) As Integer

    <PreserveSig()> _
    Function SetMaxGraphLatency(<[In]()> ByVal rtMaxGraphLatency As Long) As Integer
End Interface

<ComImport(), System.Security.SuppressUnmanagedCodeSecurity(), Guid("56a868a9-0ad4-11ce-b03a-0020af0ba770"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)> _
Public Interface IGraphBuilder
    Inherits IFilterGraph

    <PreserveSig()> _
    Shadows Function AddFilter(<[In]()> ByVal pFilter As IBaseFilter, <[In](), MarshalAs(UnmanagedType.LPWStr)> ByVal pName As String) As Integer

    <PreserveSig()> _
    Shadows Function RemoveFilter(<[In]()> ByVal pFilter As IBaseFilter) As Integer

    <PreserveSig()> _
    Shadows Function EnumFilters(<Out()> ByRef ppEnum As IEnumFilters) As Integer

    <PreserveSig()> _
    Shadows Function FindFilterByName(<[In](), MarshalAs(UnmanagedType.LPWStr)> ByVal pName As String, <Out()> ByRef ppFilter As IBaseFilter) As Integer

    <PreserveSig()> _
    Shadows Function ConnectDirect(<[In]()> ByVal ppinOut As IPin, <[In]()> ByVal ppinIn As IPin, <[In](), MarshalAs(UnmanagedType.LPStruct)> ByVal pmt As AMMediaType) As Integer

    <PreserveSig()> _
    Shadows Function Reconnect(<[In]()> ByVal ppin As IPin) As Integer

    <PreserveSig()> _
    Shadows Function Disconnect(<[In]()> ByVal ppin As IPin) As Integer

    <PreserveSig()> _
    Shadows Function SetDefaultSyncSource() As Integer

    <PreserveSig()> _
    Function Connect(<[In]()> ByVal ppinOut As IPin, <[In]()> ByVal ppinIn As IPin) As Integer

    <PreserveSig()> _
    Function Render(<[In]()> ByVal ppinOut As IPin) As Integer

    <PreserveSig()> _
    Function RenderFile(<[In](), MarshalAs(UnmanagedType.LPWStr)> ByVal lpcwstrFile As String, <[In](), MarshalAs(UnmanagedType.LPWStr)> ByVal lpcwstrPlayList As String) As Integer

    <PreserveSig()> _
    Function AddSourceFilter(<[In](), MarshalAs(UnmanagedType.LPWStr)> ByVal lpcwstrFileName As String, <[In](), MarshalAs(UnmanagedType.LPWStr)> ByVal lpcwstrFilterName As String, <Out()> ByRef ppFilter As IBaseFilter) As Integer

    <PreserveSig()> _
    Function SetLogFile(ByVal hFile As IntPtr) As Integer

    ' DWORD_PTR
    <PreserveSig()> _
    Function Abort() As Integer

    <PreserveSig()> _
    Function ShouldOperationContinue() As Integer
End Interface

<ComImport(), System.Security.SuppressUnmanagedCodeSecurity(), Guid("36b73882-c2c8-11cf-8b46-00805f6cef60"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)> _
Public Interface IFilterGraph2
    Inherits IGraphBuilder

    <PreserveSig()> _
    Shadows Function AddFilter(<[In]()> ByVal pFilter As IBaseFilter, <[In](), MarshalAs(UnmanagedType.LPWStr)> ByVal pName As String) As Integer

    <PreserveSig()> _
    Shadows Function RemoveFilter(<[In]()> ByVal pFilter As IBaseFilter) As Integer

    <PreserveSig()> _
    Shadows Function EnumFilters(<Out()> ByRef ppEnum As IEnumFilters) As Integer

    <PreserveSig()> _
    Shadows Function FindFilterByName(<[In](), MarshalAs(UnmanagedType.LPWStr)> ByVal pName As String, <Out()> ByRef ppFilter As IBaseFilter) As Integer

    <PreserveSig()> _
    Shadows Function ConnectDirect(<[In]()> ByVal ppinOut As IPin, <[In]()> ByVal ppinIn As IPin, <[In](), MarshalAs(UnmanagedType.LPStruct)> ByVal pmt As AMMediaType) As Integer

    <PreserveSig()> _
    Shadows Function Reconnect(<[In]()> ByVal ppin As IPin) As Integer

    <PreserveSig()> _
    Shadows Function Disconnect(<[In]()> ByVal ppin As IPin) As Integer

    <PreserveSig()> _
    Shadows Function SetDefaultSyncSource() As Integer

    <PreserveSig()> _
    Shadows Function Connect(<[In]()> ByVal ppinOut As IPin, <[In]()> ByVal ppinIn As IPin) As Integer

    <PreserveSig()> _
    Shadows Function Render(<[In]()> ByVal ppinOut As IPin) As Integer

    <PreserveSig()> _
    Shadows Function RenderFile(<[In](), MarshalAs(UnmanagedType.LPWStr)> ByVal lpcwstrFile As String, <[In](), MarshalAs(UnmanagedType.LPWStr)> ByVal lpcwstrPlayList As String) As Integer

    <PreserveSig()> _
    Shadows Function AddSourceFilter(<[In](), MarshalAs(UnmanagedType.LPWStr)> ByVal lpcwstrFileName As String, <[In](), MarshalAs(UnmanagedType.LPWStr)> ByVal lpcwstrFilterName As String, <Out()> ByRef ppFilter As IBaseFilter) As Integer

    <PreserveSig()> _
    Shadows Function SetLogFile(ByVal hFile As IntPtr) As Integer
    ' DWORD_PTR
    <PreserveSig()> _
    Shadows Function Abort() As Integer

    <PreserveSig()> _
    Shadows Function ShouldOperationContinue() As Integer

    <PreserveSig()> _
    Function AddSourceFilterForMoniker(<[In]()> ByVal pMoniker As IMoniker, <[In]()> ByVal pCtx As IBindCtx, <[In](), MarshalAs(UnmanagedType.LPWStr)> ByVal lpcwstrFilterName As String, <Out()> ByRef ppFilter As IBaseFilter) As Integer

    <PreserveSig()> _
    Function ReconnectEx(<[In]()> ByVal ppin As IPin, <[In]()> ByVal pmt As AMMediaType) As Integer

    ' DWORD *
    <PreserveSig()> _
    Function RenderEx(<[In]()> ByVal pPinOut As IPin, <[In]()> ByVal dwFlags As AMRenderExFlags, <[In]()> ByVal pvContext As IntPtr) As Integer
End Interface

<ComImport(), System.Security.SuppressUnmanagedCodeSecurity(), Guid("5ACD6AA0-F482-11ce-8B67-00AA00A3F1A6"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)> _
Public Interface IConfigAviMux
    <PreserveSig()> _
    Function SetMasterStream(<[In]()> ByVal iStream As Integer) As Integer

    <PreserveSig()> _
    Function GetMasterStream(<Out()> ByRef pStream As Integer) As Integer

    <PreserveSig()> _
    Function SetOutputCompatibilityIndex(<[In](), MarshalAs(UnmanagedType.Bool)> ByVal fOldIndex As Boolean) As Integer

    <PreserveSig()> _
    Function GetOutputCompatibilityIndex(<Out(), MarshalAs(UnmanagedType.Bool)> ByRef pfOldIndex As Boolean) As Integer
End Interface

<ComImport(), System.Security.SuppressUnmanagedCodeSecurity(), Guid("BEE3D220-157B-11d0-BD23-00A0C911CE86"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)> _
Public Interface IConfigInterleaving
    <PreserveSig()> _
    Function put_Mode(<[In]()> ByVal mode As InterleavingMode) As Integer

    <PreserveSig()> _
    Function get_Mode(<Out()> ByRef pMode As InterleavingMode) As Integer

    <PreserveSig()> _
    Function put_Interleaving(<[In]()> ByRef prtInterleave As Long, <[In]()> ByRef prtPreroll As Long) As Integer

    <PreserveSig()> _
    Function get_Interleaving(<Out()> ByRef prtInterleave As Long, <Out()> ByRef prtPreroll As Long) As Integer
End Interface

<ComImport(), System.Security.SuppressUnmanagedCodeSecurity(), Guid("a2104830-7c70-11cf-8bce-00aa00a3f1a6"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)> _
Public Interface IFileSinkFilter
    <PreserveSig()> _
    Function SetFileName(<[In](), MarshalAs(UnmanagedType.LPWStr)> ByVal pszFileName As String, <[In](), MarshalAs(UnmanagedType.LPStruct)> ByVal pmt As AMMediaType) As Integer

    <PreserveSig()> _
    Function GetCurFile(<Out(), MarshalAs(UnmanagedType.LPWStr)> ByRef pszFileName As String, <Out(), MarshalAs(UnmanagedType.LPStruct)> ByVal pmt As AMMediaType) As Integer
End Interface

<ComImport(), System.Security.SuppressUnmanagedCodeSecurity(), Guid("00855B90-CE1B-11d0-BD4F-00A0C911CE86"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)> _
Public Interface IFileSinkFilter2
    Inherits IFileSinkFilter

    <PreserveSig()> _
    Shadows Function SetFileName(<[In](), MarshalAs(UnmanagedType.LPWStr)> ByVal pszFileName As String, <[In](), MarshalAs(UnmanagedType.LPStruct)> ByVal pmt As AMMediaType) As Integer

    <PreserveSig()> _
    Shadows Function GetCurFile(<Out(), MarshalAs(UnmanagedType.LPWStr)> ByRef pszFileName As String, <Out(), MarshalAs(UnmanagedType.LPStruct)> ByVal pmt As AMMediaType) As Integer

    <PreserveSig()> _
    Function SetMode(<[In]()> ByVal dwFlags As AMFileSinkFlags) As Integer

    <PreserveSig()> _
    Function GetMode(<Out()> ByRef dwFlags As AMFileSinkFlags) As Integer
End Interface

<ComImport(), System.Security.SuppressUnmanagedCodeSecurity(), Guid("56a868a6-0ad4-11ce-b03a-0020af0ba770"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)> _
Public Interface IFileSourceFilter
    <PreserveSig()> _
    Function Load(<[In](), MarshalAs(UnmanagedType.LPWStr)> ByVal pszFileName As String, <[In](), MarshalAs(UnmanagedType.LPStruct)> ByVal pmt As AMMediaType) As Integer

    <PreserveSig()> _
    Function GetCurFile(<Out(), MarshalAs(UnmanagedType.LPWStr)> ByRef pszFileName As String, <Out(), MarshalAs(UnmanagedType.LPStruct)> ByVal pmt As AMMediaType) As Integer
End Interface

<ComImport(), System.Security.SuppressUnmanagedCodeSecurity(), Guid("e46a9787-2b71-444d-a4b5-1fab7b708d6a"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)> _
Public Interface IVideoFrameStep
    <PreserveSig()> _
    Function [Step](<[In]()> ByVal dwFrames As Integer, <[In](), MarshalAs(UnmanagedType.IUnknown)> ByVal pStepObject As Object) As Integer

    <PreserveSig()> _
    Function CanStep(<[In]()> ByVal bMultiple As Integer, <[In](), MarshalAs(UnmanagedType.IUnknown)> ByVal pStepObject As Object) As Integer

    <PreserveSig()> _
    Function CancelStep() As Integer
End Interface

<ComImport(), System.Security.SuppressUnmanagedCodeSecurity(), Guid("31EFAC30-515C-11d0-A9AA-00AA0061BE93"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)> _
Public Interface IKsPropertySet
    <PreserveSig()> _
    Function [Set](<[In](), MarshalAs(UnmanagedType.LPStruct)> ByVal guidPropSet As Guid, <[In]()> ByVal dwPropID As Integer, <[In]()> ByVal pInstanceData As IntPtr, <[In]()> ByVal cbInstanceData As Integer, <[In]()> ByVal pPropData As IntPtr, <[In]()> ByVal cbPropData As Integer) As Integer

    <PreserveSig()> _
    Function [Get](<[In](), MarshalAs(UnmanagedType.LPStruct)> ByVal guidPropSet As Guid, <[In]()> ByVal dwPropID As Integer, <[In]()> ByVal pInstanceData As IntPtr, <[In]()> ByVal cbInstanceData As Integer, <[In](), Out()> ByVal pPropData As IntPtr, <[In]()> ByVal cbPropData As Integer, _
     <Out()> ByRef pcbReturned As Integer) As Integer

    <PreserveSig()> _
    Function QuerySupported(<[In](), MarshalAs(UnmanagedType.LPStruct)> ByVal guidPropSet As Guid, <[In]()> ByVal dwPropID As Integer, <Out()> ByRef pTypeSupport As KSPropertySupport) As Integer
End Interface

<ComImport(), System.Security.SuppressUnmanagedCodeSecurity(), Guid("211A8761-03AC-11d1-8D13-00AA00BD8339"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)> _
Public Interface IAMTuner
    <PreserveSig()> _
    Function put_Channel(<[In]()> ByVal lChannel As Integer, <[In]()> ByVal lVideoSubChannel As AMTunerSubChannel, <[In]()> ByVal lAudioSubChannel As AMTunerSubChannel) As Integer

    <PreserveSig()> _
    Function get_Channel(<Out()> ByRef plChannel As Integer, <Out()> ByRef plVideoSubChannel As AMTunerSubChannel, <Out()> ByRef plAudioSubChannel As AMTunerSubChannel) As Integer

    <PreserveSig()> _
    Function ChannelMinMax(<Out()> ByRef lChannelMin As Integer, <Out()> ByRef lChannelMax As Integer) As Integer

    <PreserveSig()> _
    Function put_CountryCode(<[In]()> ByVal lCountryCode As Integer) As Integer

    <PreserveSig()> _
    Function get_CountryCode(<Out()> ByRef plCountryCode As Integer) As Integer

    <PreserveSig()> _
    Function put_TuningSpace(<[In]()> ByVal lTuningSpace As Integer) As Integer

    <PreserveSig()> _
    Function get_TuningSpace(<Out()> ByRef plTuningSpace As Integer) As Integer

    <PreserveSig()> _
    Function Logon(<[In]()> ByVal hCurrentUser As IntPtr) As Integer

    ' HANDLE
    <PreserveSig()> _
    Function Logout() As Integer

    <PreserveSig()> _
    Function SignalPresent(<Out()> ByRef plSignalStrength As AMTunerSignalStrength) As Integer

    <PreserveSig()> _
    Function put_Mode(<[In]()> ByVal lMode As AMTunerModeType) As Integer

    <PreserveSig()> _
    Function get_Mode(<Out()> ByRef plMode As AMTunerModeType) As Integer

    <PreserveSig()> _
    Function GetAvailableModes(<Out()> ByRef plModes As AMTunerModeType) As Integer

    <PreserveSig()> _
    Function RegisterNotificationCallBack(<[In]()> ByVal pNotify As IAMTunerNotification, <[In]()> ByVal lEvents As AMTunerEventType) As Integer

    <PreserveSig()> _
    Function UnRegisterNotificationCallBack(<[In]()> ByVal pNotify As IAMTunerNotification) As Integer
End Interface

<ComImport(), System.Security.SuppressUnmanagedCodeSecurity(), Guid("211A8760-03AC-11d1-8D13-00AA00BD8339"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)> _
Public Interface IAMTunerNotification
    <PreserveSig()> _
    Function OnEvent(<[In]()> ByVal [Event] As AMTunerEventType) As Integer
End Interface

<ComImport(), System.Security.SuppressUnmanagedCodeSecurity(), Guid("211A8766-03AC-11d1-8D13-00AA00BD8339"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)> _
Public Interface IAMTVTuner
    Inherits IAMTuner

    <PreserveSig()> _
    Shadows Function put_Channel(<[In]()> ByVal lChannel As Integer, <[In]()> ByVal lVideoSubChannel As AMTunerSubChannel, <[In]()> ByVal lAudioSubChannel As AMTunerSubChannel) As Integer

    <PreserveSig()> _
    Shadows Function get_Channel(<Out()> ByRef plChannel As Integer, <Out()> ByRef plVideoSubChannel As AMTunerSubChannel, <Out()> ByRef plAudioSubChannel As AMTunerSubChannel) As Integer

    <PreserveSig()> _
    Shadows Function ChannelMinMax(<Out()> ByRef lChannelMin As Integer, <Out()> ByRef lChannelMax As Integer) As Integer

    <PreserveSig()> _
    Shadows Function put_CountryCode(<[In]()> ByVal lCountryCode As Integer) As Integer

    <PreserveSig()> _
    Shadows Function get_CountryCode(<Out()> ByRef plCountryCode As Integer) As Integer

    <PreserveSig()> _
    Shadows Function put_TuningSpace(<[In]()> ByVal lTuningSpace As Integer) As Integer

    <PreserveSig()> _
    Shadows Function get_TuningSpace(<Out()> ByRef plTuningSpace As Integer) As Integer

    <PreserveSig()> _
    Shadows Function Logon(<[In]()> ByVal hCurrentUser As IntPtr) As Integer

    ' HANDLE
    <PreserveSig()> _
    Shadows Function Logout() As Integer

    <PreserveSig()> _
    Shadows Function SignalPresent(<Out()> ByRef plSignalStrength As AMTunerSignalStrength) As Integer

    <PreserveSig()> _
    Shadows Function put_Mode(<[In]()> ByVal lMode As AMTunerModeType) As Integer

    <PreserveSig()> _
    Shadows Function get_Mode(<Out()> ByRef plMode As AMTunerModeType) As Integer

    <PreserveSig()> _
    Shadows Function GetAvailableModes(<Out()> ByRef plModes As AMTunerModeType) As Integer

    <PreserveSig()> _
    Shadows Function RegisterNotificationCallBack(<[In]()> ByVal pNotify As IAMTunerNotification, <[In]()> ByVal lEvents As AMTunerEventType) As Integer

    <PreserveSig()> _
    Shadows Function UnRegisterNotificationCallBack(<[In]()> ByVal pNotify As IAMTunerNotification) As Integer

    <PreserveSig()> _
    Function get_AvailableTVFormats(<Out()> ByRef lAnalogVideoStandard As AnalogVideoStandard) As Integer

    <PreserveSig()> _
    Function get_TVFormat(<Out()> ByRef lAnalogVideoStandard As AnalogVideoStandard) As Integer

    <PreserveSig()> _
    Function AutoTune(<[In]()> ByVal lChannel As Integer, <Out()> ByRef plFoundSignal As Integer) As Integer

    <PreserveSig()> _
    Function StoreAutoTune() As Integer

    <PreserveSig()> _
    Function get_NumInputConnections(<Out()> ByRef plNumInputConnections As Integer) As Integer

    <PreserveSig()> _
    Function put_InputType(<[In]()> ByVal lIndex As Integer, <[In]()> ByVal inputType As TunerInputType) As Integer

    <PreserveSig()> _
    Function get_InputType(<[In]()> ByVal lIndex As Integer, <Out()> ByRef inputType As TunerInputType) As Integer

    <PreserveSig()> _
    Function put_ConnectInput(<[In]()> ByVal lIndex As Integer) As Integer

    <PreserveSig()> _
    Function get_ConnectInput(<Out()> ByRef lIndex As Integer) As Integer

    <PreserveSig()> _
    Function get_VideoFrequency(<Out()> ByRef lFreq As Integer) As Integer

    <PreserveSig()> _
    Function get_AudioFrequency(<Out()> ByRef lFreq As Integer) As Integer
End Interface

<ComImport(), System.Security.SuppressUnmanagedCodeSecurity(), Guid("6a2e0670-28e4-11d0-a18c-00a0c9118956"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)> _
Public Interface IAMVideoControl
    <PreserveSig()> _
    Function GetCaps(<[In]()> ByVal pPin As IPin, <Out()> ByRef pCapsFlags As VideoControlFlags) As Integer

    <PreserveSig()> _
    Function SetMode(<[In]()> ByVal pPin As IPin, <[In]()> ByVal Mode As VideoControlFlags) As Integer

    <PreserveSig()> _
    Function GetMode(<[In]()> ByVal pPin As IPin, <Out()> ByRef Mode As VideoControlFlags) As Integer

    <PreserveSig()> _
    Function GetCurrentActualFrameRate(<[In]()> ByVal pPin As IPin, <Out()> ByRef ActualFrameRate As Long) As Integer

    <PreserveSig()> _
    Function GetMaxAvailableFrameRate(<[In]()> ByVal pPin As IPin, <[In]()> ByVal iIndex As Integer, <[In]()> ByVal Dimensions As Size, <Out()> ByRef MaxAvailableFrameRate As Long) As Integer

    <PreserveSig()> _
    Function GetFrameRateList(<[In]()> ByVal pPin As IPin, <[In]()> ByVal iIndex As Integer, <[In]()> ByVal Dimensions As Size, <Out()> ByRef ListSize As Integer, <Out()> ByRef FrameRates As IntPtr) As Integer
End Interface

<ComImport(), System.Security.SuppressUnmanagedCodeSecurity(), Guid("C6E13350-30AC-11d0-A18C-00A0C9118956"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)> _
Public Interface IAMAnalogVideoDecoder
    <PreserveSig()> _
    Function get_AvailableTVFormats(<Out()> ByRef lAnalogVideoStandard As AnalogVideoStandard) As Integer

    <PreserveSig()> _
    Function put_TVFormat(<[In]()> ByVal lAnalogVideoStandard As AnalogVideoStandard) As Integer

    <PreserveSig()> _
    Function get_TVFormat(<Out()> ByRef plAnalogVideoStandard As AnalogVideoStandard) As Integer

    <PreserveSig()> _
    Function get_HorizontalLocked(<Out(), MarshalAs(UnmanagedType.Bool)> ByRef plLocked As Boolean) As Integer

    <PreserveSig()> _
    Function put_VCRHorizontalLocking(<[In](), MarshalAs(UnmanagedType.Bool)> ByVal lVCRHorizontalLocking As Boolean) As Integer

    <PreserveSig()> _
    Function get_VCRHorizontalLocking(<Out(), MarshalAs(UnmanagedType.Bool)> ByRef plVCRHorizontalLocking As Boolean) As Integer

    <PreserveSig()> _
    Function get_NumberOfLines(<Out()> ByRef plNumberOfLines As Integer) As Integer

    <PreserveSig()> _
    Function put_OutputEnable(<[In](), MarshalAs(UnmanagedType.Bool)> ByVal lOutputEnable As Boolean) As Integer

    <PreserveSig()> _
    Function get_OutputEnable(<Out(), MarshalAs(UnmanagedType.Bool)> ByRef plOutputEnable As Boolean) As Integer
End Interface

<ComImport(), System.Security.SuppressUnmanagedCodeSecurity(), Guid("C6E13360-30AC-11d0-A18C-00A0C9118956"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)> _
Public Interface IAMVideoProcAmp
    <PreserveSig()> _
    Function GetRange(<[In]()> ByVal [Property] As VideoProcAmpProperty, <Out()> ByRef pMin As Integer, <Out()> ByRef pMax As Integer, <Out()> ByRef pSteppingDelta As Integer, <Out()> ByRef pDefault As Integer, <Out()> ByRef pCapsFlags As VideoProcAmpFlags) As Integer

    <PreserveSig()> _
    Function [Set](<[In]()> ByVal [Property] As VideoProcAmpProperty, <[In]()> ByVal lValue As Integer, <[In]()> ByVal Flags As VideoProcAmpFlags) As Integer

    <PreserveSig()> _
    Function [Get](<[In]()> ByVal [Property] As VideoProcAmpProperty, <Out()> ByRef lValue As Integer, <Out()> ByRef Flags As VideoProcAmpFlags) As Integer
End Interface

<ComImport(), System.Security.SuppressUnmanagedCodeSecurity(), Guid("54C39221-8380-11d0-B3F0-00AA003761C5"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)> _
Public Interface IAMAudioInputMixer
    <PreserveSig()> _
    Function put_Enable(<[In](), MarshalAs(UnmanagedType.Bool)> ByVal fEnable As Boolean) As Integer

    <PreserveSig()> _
    Function get_Enable(<Out(), MarshalAs(UnmanagedType.Bool)> ByRef pfEnable As Boolean) As Integer

    <PreserveSig()> _
    Function put_Mono(<[In](), MarshalAs(UnmanagedType.Bool)> ByVal fMono As Boolean) As Integer

    <PreserveSig()> _
    Function get_Mono(<Out(), MarshalAs(UnmanagedType.Bool)> ByRef pfMono As Boolean) As Integer

    <PreserveSig()> _
    Function put_MixLevel(<[In]()> ByVal Level As Double) As Integer

    <PreserveSig()> _
    Function get_MixLevel(<Out()> ByRef pLevel As Double) As Integer

    <PreserveSig()> _
    Function put_Pan(<[In]()> ByVal Pan As Double) As Integer

    <PreserveSig()> _
    Function get_Pan(<Out()> ByRef pPan As Double) As Integer

    <PreserveSig()> _
    Function put_Loudness(<[In](), MarshalAs(UnmanagedType.Bool)> ByVal fLoudness As Boolean) As Integer

    <PreserveSig()> _
    Function get_Loudness(<Out(), MarshalAs(UnmanagedType.Bool)> ByRef pfLoudness As Boolean) As Integer

    <PreserveSig()> _
    Function put_Treble(<[In]()> ByVal Treble As Double) As Integer

    <PreserveSig()> _
    Function get_Treble(<Out()> ByRef pTreble As Double) As Integer

    <PreserveSig()> _
    Function get_TrebleRange(<Out()> ByRef pRange As Double) As Integer

    <PreserveSig()> _
    Function put_Bass(<[In]()> ByVal Bass As Double) As Integer

    <PreserveSig()> _
    Function get_Bass(<Out()> ByRef pBass As Double) As Integer

    <PreserveSig()> _
    Function get_BassRange(<Out()> ByRef pRange As Double) As Integer
End Interface

<ComImport(), System.Security.SuppressUnmanagedCodeSecurity(), Guid("670d1d20-a068-11d0-b3f0-00aa003761c5"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)> _
Public Interface IAMCopyCaptureFileProgress
    <PreserveSig()> _
    Function Progress(ByVal iProgress As Integer) As Integer
End Interface

<ComImport(), System.Security.SuppressUnmanagedCodeSecurity(), Guid("C6E13380-30AC-11d0-A18C-00A0C9118956"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)> _
Public Interface IAMCrossbar
    <PreserveSig()> _
    Function get_PinCounts(<Out()> ByRef OutputPinCount As Integer, <Out()> ByRef InputPinCount As Integer) As Integer

    <PreserveSig()> _
    Function CanRoute(<[In]()> ByVal OutputPinIndex As Integer, <[In]()> ByVal InputPinIndex As Integer) As Integer

    <PreserveSig()> _
    Function Route(<[In]()> ByVal OutputPinIndex As Integer, <[In]()> ByVal InputPinIndex As Integer) As Integer

    <PreserveSig()> _
    Function get_IsRoutedTo(<[In]()> ByVal OutputPinIndex As Integer, <Out()> ByRef InputPinIndex As Integer) As Integer

    <PreserveSig()> _
    Function get_CrossbarPinInfo(<[In](), MarshalAs(UnmanagedType.Bool)> ByVal IsInputPin As Boolean, <[In]()> ByVal PinIndex As Integer, <Out()> ByRef PinIndexRelated As Integer, <Out()> ByRef PhysicalType As PhysicalConnectorType) As Integer
End Interface

<ComImport(), System.Security.SuppressUnmanagedCodeSecurity(), Guid("C6E13344-30AC-11d0-A18C-00A0C9118956"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)> _
Public Interface IAMDroppedFrames
    <PreserveSig()> _
    Function GetNumDropped(<Out()> ByRef plDropped As Integer) As Integer

    <PreserveSig()> _
    Function GetNumNotDropped(<Out()> ByRef plNotDropped As Integer) As Integer

    <PreserveSig()> _
    Function GetDroppedInfo(<[In]()> ByVal lSize As Integer, <Out(), MarshalAs(UnmanagedType.LPArray)> ByRef plArray As Integer(), <Out()> ByRef plNumCopied As Integer) As Integer

    <PreserveSig()> _
    Function GetAverageFrameSize(<Out()> ByRef plAverageSize As Integer) As Integer
End Interface

<ComImport(), System.Security.SuppressUnmanagedCodeSecurity(), Guid("83EC1C30-23D1-11d1-99E6-00A0C9560266"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)> _
Public Interface IAMTVAudio
    <PreserveSig()> _
    Function GetHardwareSupportedTVAudioModes(<Out()> ByRef plModes As TVAudioMode) As Integer

    <PreserveSig()> _
    Function GetAvailableTVAudioModes(<Out()> ByRef plModes As TVAudioMode) As Integer

    <PreserveSig()> _
    Function get_TVAudioMode(<Out()> ByRef plMode As TVAudioMode) As Integer

    <PreserveSig()> _
    Function put_TVAudioMode(<[In]()> ByVal lMode As TVAudioMode) As Integer

    <PreserveSig()> _
    Function RegisterNotificationCallBack(<[In]()> ByVal pNotify As IAMTunerNotification, <[In]()> ByVal lEvents As AMTVAudioEventType) As Integer

    <PreserveSig()> _
    Function UnRegisterNotificationCallBack(<[In]()> ByVal pNotify As IAMTunerNotification) As Integer
End Interface

<ComImport(), System.Security.SuppressUnmanagedCodeSecurity(), Guid("D8D715A3-6E5E-11D0-B3F0-00AA003761C5"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)> _
Public Interface IAMVfwCompressDialogs
    <PreserveSig()> _
    Function ShowDialog(<[In]()> ByVal iDialog As VfwCompressDialogs, <[In]()> ByVal hwnd As IntPtr) As Integer

    <PreserveSig()> _
    Function GetState(<[In]()> ByVal pState As IntPtr, <[In](), Out()> ByRef pcbState As Integer) As Integer

    <PreserveSig()> _
    Function SetState(<[In]()> ByVal pState As IntPtr, <[In]()> ByVal pcbState As Integer) As Integer

    <PreserveSig()> _
    Function SendDriverMessage(<[In]()> ByVal uMsg As Integer, <[In]()> ByVal dw1 As Integer, <[In]()> ByVal dw2 As Integer) As Integer
End Interface

<ComImport(), System.Security.SuppressUnmanagedCodeSecurity(), Guid("C6E13343-30AC-11d0-A18C-00A0C9118956"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)> _
Public Interface IAMVideoCompression
    <PreserveSig()> _
    Function put_KeyFrameRate(<[In]()> ByVal KeyFrameRate As Integer) As Integer

    <PreserveSig()> _
    Function get_KeyFrameRate(<Out()> ByRef pKeyFrameRate As Integer) As Integer

    <PreserveSig()> _
    Function put_PFramesPerKeyFrame(<[In]()> ByVal PFramesPerKeyFrame As Integer) As Integer

    <PreserveSig()> _
    Function get_PFramesPerKeyFrame(<Out()> ByRef pPFramesPerKeyFrame As Integer) As Integer

    <PreserveSig()> _
    Function put_Quality(<[In]()> ByVal Quality As Double) As Integer

    <PreserveSig()> _
    Function get_Quality(<Out()> ByRef pQuality As Double) As Integer

    <PreserveSig()> _
    Function put_WindowSize(<[In]()> ByVal WindowSize As Long) As Integer

    <PreserveSig()> _
    Function get_WindowSize(<Out()> ByRef pWindowSize As Long) As Integer

    ' WCHAR *
    ' LPWSTR
    <PreserveSig()> _
    Function GetInfo(<MarshalAs(UnmanagedType.LPWStr)> ByVal pszVersion As StringBuilder, <Out()> ByRef pcbVersion As Integer, <MarshalAs(UnmanagedType.LPWStr)> ByVal pszDescription As StringBuilder, <Out()> ByRef pcbDescription As Integer, <Out()> ByRef pDefaultKeyFrameRate As Integer, <Out()> ByRef pDefaultPFramesPerKey As Integer, _
     <Out()> ByRef pDefaultQuality As Double, <Out()> ByRef pCapabilities As CompressionCaps) As Integer

    <PreserveSig()> _
    Function OverrideKeyFrame(<[In]()> ByVal FrameNumber As Integer) As Integer

    <PreserveSig()> _
    Function OverrideFrameSize(<[In]()> ByVal FrameNumber As Integer, <[In]()> ByVal Size As Integer) As Integer
End Interface

<ComImport(), System.Security.SuppressUnmanagedCodeSecurity(), Guid("93E5A4E0-2D50-11d2-ABFA-00A0C9C6E38D"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)> _
Public Interface ICaptureGraphBuilder2
    <PreserveSig()> _
    Function SetFiltergraph(<[In]()> ByVal pfg As IGraphBuilder) As Integer

    <PreserveSig()> _
    Function GetFiltergraph(<Out()> ByRef ppfg As IGraphBuilder) As Integer

    <PreserveSig()> _
    Function SetOutputFileName(<[In](), MarshalAs(UnmanagedType.LPStruct)> ByVal pType As Guid, <[In](), MarshalAs(UnmanagedType.LPWStr)> ByVal lpstrFile As String, <Out()> ByRef ppbf As IBaseFilter, <Out()> ByRef ppSink As IFileSinkFilter) As Integer

    <PreserveSig()> _
    Function FindInterface(<[In](), MarshalAs(UnmanagedType.LPStruct)> ByVal pCategory As DsGuid, <[In](), MarshalAs(UnmanagedType.LPStruct)> ByVal pType As DsGuid, <[In]()> ByVal pbf As IBaseFilter, <[In](), MarshalAs(UnmanagedType.LPStruct)> ByVal riid As Guid, <Out(), MarshalAs(UnmanagedType.IUnknown)> ByRef ppint As Object) As Integer

    <PreserveSig()> _
    Function RenderStream(<[In](), MarshalAs(UnmanagedType.LPStruct)> ByVal PinCategory As DsGuid, <[In](), MarshalAs(UnmanagedType.LPStruct)> ByVal MediaType As DsGuid, <[In](), MarshalAs(UnmanagedType.IUnknown)> ByVal pSource As Object, <[In]()> ByVal pfCompressor As IBaseFilter, <[In]()> ByVal pfRenderer As IBaseFilter) As Integer

    <PreserveSig()> _
    Function ControlStream(<[In](), MarshalAs(UnmanagedType.LPStruct)> ByVal PinCategory As Guid, <[In](), MarshalAs(UnmanagedType.LPStruct)> ByVal MediaType As DsGuid, <[In](), MarshalAs(UnmanagedType.[Interface])> ByVal pFilter As IBaseFilter, <[In]()> ByVal pstart As DsLong, <[In]()> ByVal pstop As DsLong, <[In]()> ByVal wStartCookie As Short, _
     <[In]()> ByVal wStopCookie As Short) As Integer

    <PreserveSig()> _
    Function AllocCapFile(<[In](), MarshalAs(UnmanagedType.LPWStr)> ByVal lpstrFile As String, <[In]()> ByVal dwlSize As Long) As Integer

    <PreserveSig()> _
    Function CopyCaptureFile(<[In](), MarshalAs(UnmanagedType.LPWStr)> ByVal lpwstrOld As String, <[In](), MarshalAs(UnmanagedType.LPWStr)> ByVal lpwstrNew As String, <[In](), MarshalAs(UnmanagedType.Bool)> ByVal fAllowEscAbort As Boolean, <[In]()> ByVal pFilter As IAMCopyCaptureFileProgress) As Integer

    <PreserveSig()> _
    Function FindPin(<[In](), MarshalAs(UnmanagedType.IUnknown)> ByVal pSource As Object, <[In]()> ByVal pindir As PinDirection, <[In](), MarshalAs(UnmanagedType.LPStruct)> ByVal PinCategory As DsGuid, <[In](), MarshalAs(UnmanagedType.LPStruct)> ByVal MediaType As DsGuid, <[In](), MarshalAs(UnmanagedType.Bool)> ByVal fUnconnected As Boolean, <[In]()> ByVal ZeroBasedIndex As Integer, _
     <Out(), MarshalAs(UnmanagedType.[Interface])> ByRef ppPin As IPin) As Integer
End Interface

<ComImport(), System.Security.SuppressUnmanagedCodeSecurity(), Guid("C6E13340-30AC-11d0-A18C-00A0C9118956"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)> _
Public Interface IAMStreamConfig
    <PreserveSig()> _
    Function SetFormat(<[In](), MarshalAs(UnmanagedType.LPStruct)> ByVal pmt As AMMediaType) As Integer

    <PreserveSig()> _
    Function GetFormat(<Out()> ByRef pmt As AMMediaType) As Integer

    <PreserveSig()> _
    Function GetNumberOfCapabilities(ByRef piCount As Integer, ByRef piSize As Integer) As Integer

    <PreserveSig()> _
    Function GetStreamCaps(<[In]()> ByVal iIndex As Integer, <Out()> ByRef ppmt As AMMediaType, <[In]()> ByVal pSCC As IntPtr) As Integer
End Interface

<ComImport(), System.Security.SuppressUnmanagedCodeSecurity(), Guid("d18e17a0-aacb-11d0-afb0-00aa00b67a42"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)> _
Public Interface IDVEnc
    <PreserveSig()> _
    Function get_IFormatResolution(<Out()> ByRef VideoFormat As DVEncoderVideoFormat, <Out()> ByRef DVFormat As DVEncoderFormat, <Out()> ByRef Resolution As DVEncoderResolution, <[In]()> ByVal fDVInfo As OABool, <Out()> ByRef sDVInfo As DVInfo) As Integer

    <PreserveSig()> _
    Function put_IFormatResolution(<[In]()> ByVal VideoFormat As DVEncoderVideoFormat, <[In]()> ByVal DVFormat As DVEncoderFormat, <[In]()> ByVal Resolution As DVEncoderResolution, <[In]()> ByVal fDVInfo As OABool, <[In]()> ByVal sDVInfo As DVInfo) As Integer
End Interface

<ComImport(), System.Security.SuppressUnmanagedCodeSecurity(), Guid("56a868a2-0ad4-11ce-b03a-0020af0ba770"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)> _
Public Interface IMediaEventSink
    <PreserveSig()> _
    Function Notify(<[In]()> ByVal evCode As EventCode, <[In]()> ByVal EventParam1 As IntPtr, <[In]()> ByVal EventParam2 As IntPtr) As Integer
End Interface

<ComImport(), System.Security.SuppressUnmanagedCodeSecurity(), Guid("9FD52741-176D-4b36-8F51-CA8F933223BE"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)> _
Public Interface IAMClockSlave
    <PreserveSig()> _
    Function SetErrorTolerance(<[In]()> ByVal dwTolerance As Integer) As Integer

    <PreserveSig()> _
    Function GetErrorTolerance(<Out()> ByRef pdwTolerance As Integer) As Integer
End Interface

<ComImport(), System.Security.SuppressUnmanagedCodeSecurity(), Guid("c0dff467-d499-4986-972b-e1d9090fa941"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)> _
Public Interface IAMDecoderCaps
    <PreserveSig()> _
    Function GetDecoderCaps(<[In]()> ByVal dwCapIndex As AMQueryDecoder, <Out()> ByRef lpdwCap As DecoderCap) As Integer
End Interface

<ComImport(), System.Security.SuppressUnmanagedCodeSecurity(), Guid("62fae250-7e65-4460-bfc9-6398b322073c"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)> _
Public Interface IAMOverlayFX
    <PreserveSig()> _
    Function QueryOverlayFXCaps(<Out()> ByRef lpdwOverlayFXCaps As AMOverlayFX) As Integer

    <PreserveSig()> _
    Function SetOverlayFX(<[In]()> ByVal dwOverlayFX As AMOverlayFX) As Integer

    <PreserveSig()> _
    Function GetOverlayFX(<Out()> ByRef lpdwOverlayFX As AMOverlayFX) As Integer
End Interface

<ComImport(), System.Security.SuppressUnmanagedCodeSecurity(), Guid("8389d2d0-77d7-11d1-abe6-00a0c905f375"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)> _
Public Interface IAMResourceControl
    ' PVOID
    <PreserveSig()> _
    Function Reserve(<[In]()> ByVal dwFlags As AMResCtlReserveFlags, <[In]()> ByVal pvReserved As IntPtr) As Integer
End Interface

<ComImport(), System.Security.SuppressUnmanagedCodeSecurity(), Guid("c1960960-17f5-11d1-abe1-00a0c905f375"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)> _
Public Interface IAMStreamSelect
    <PreserveSig()> _
    Function Count(<Out()> ByRef pcStreams As Integer) As Integer

    <PreserveSig()> _
    Function Info(<[In]()> ByVal lIndex As Integer, <Out()> ByRef ppmt As AMMediaType, <Out()> ByRef pdwFlags As AMStreamSelectInfoFlags, <Out()> ByRef plcid As Integer, <Out()> ByRef pdwGroup As Integer, <Out(), MarshalAs(UnmanagedType.LPWStr)> ByRef ppszName As String, _
     <Out(), MarshalAs(UnmanagedType.IUnknown)> ByRef ppObject As Object, <Out(), MarshalAs(UnmanagedType.IUnknown)> ByRef ppUnk As Object) As Integer

    <PreserveSig()> _
    Function Enable(<[In]()> ByVal lIndex As Integer, <[In]()> ByVal dwFlags As AMStreamSelectEnableFlags) As Integer
End Interface

<ComImport(), System.Security.SuppressUnmanagedCodeSecurity(), Guid("60d32930-13da-11d3-9ec6-c4fcaef5c7be"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)> _
Public Interface IAMVideoDecimationProperties
    <PreserveSig()> _
    Function QueryDecimationUsage(<Out()> ByRef lpUsage As DecimationUsage) As Integer

    <PreserveSig()> _
    Function SetDecimationUsage(<[In]()> ByVal Usage As DecimationUsage) As Integer
End Interface

<ComImport(), System.Security.SuppressUnmanagedCodeSecurity(), Guid("b79bb0b0-33c1-11d1-abe1-00a0c905f375"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)> _
Public Interface IFilterMapper2
    <PreserveSig()> _
    Function CreateCategory(<[In](), MarshalAs(UnmanagedType.LPStruct)> ByVal clsidCategory As Guid, <[In]()> ByVal dwCategoryMerit As Merit, <[In](), MarshalAs(UnmanagedType.LPWStr)> ByVal Description As String) As Integer

    <PreserveSig()> _
    Function UnregisterFilter(<[In](), MarshalAs(UnmanagedType.LPStruct)> ByVal clsidCategory As Guid, <[In](), MarshalAs(UnmanagedType.LPWStr)> ByVal szInstance As String, <[In](), MarshalAs(UnmanagedType.LPStruct)> ByVal Filter As Guid) As Integer



    <PreserveSig()> _
    <Obsolete("This method has not been tested.", False)> _
    Function RegisterFilter(<[In]()> ByVal clsidFilter As Guid, <[In](), MarshalAs(UnmanagedType.LPWStr)> ByVal Name As String, <[In](), Out()> ByVal ppMoniker As IMoniker, <[In]()> ByVal pclsidCategory As DsGuid, <[In](), MarshalAs(UnmanagedType.LPWStr)> ByVal szInstance As String, <[In]()> ByVal prf2 As Object) As Integer

    ' GUID *
    ' GUID *
    <PreserveSig()> _
    Function EnumMatchingFilters(<Out()> ByRef ppEnum As IEnumMoniker, <[In]()> ByVal dwFlags As Integer, <[In](), MarshalAs(UnmanagedType.Bool)> ByVal bExactMatch As Boolean, <[In]()> ByVal dwMerit As Merit, <[In](), MarshalAs(UnmanagedType.Bool)> ByVal bInputNeeded As Boolean, <[In]()> ByVal cInputTypes As Integer, _
     <[In](), MarshalAs(UnmanagedType.LPArray, ArraySubType:=UnmanagedType.Struct)> ByVal pInputTypes As Guid(), <[In]()> ByVal pMedIn As RegPinMedium, <[In]()> ByVal pPinCategoryIn As DsGuid, <[In](), MarshalAs(UnmanagedType.Bool)> ByVal bRender As Boolean, <[In](), MarshalAs(UnmanagedType.Bool)> ByVal bOutputNeeded As Boolean, <[In]()> ByVal cOutputTypes As Integer, _
     <[In](), MarshalAs(UnmanagedType.LPArray, ArraySubType:=UnmanagedType.Struct)> ByVal pOutputTypes As Guid(), <[In]()> ByVal pMedOut As RegPinMedium, <[In]()> ByVal pPinCategoryOut As DsGuid) As Integer
End Interface

<ComImport(), System.Security.SuppressUnmanagedCodeSecurity(), Guid("b79bb0b1-33c1-11d1-abe1-00a0c905f375"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)> _
Public Interface IFilterMapper3
    Inherits IFilterMapper2

    <PreserveSig()> _
    Shadows Function CreateCategory(<[In]()> ByVal clsidCategory As Guid, <[In]()> ByVal dwCategoryMerit As Merit, <[In](), MarshalAs(UnmanagedType.LPWStr)> ByVal Description As String) As Integer

    <PreserveSig()> _
    Shadows Function UnregisterFilter(<[In]()> ByVal clsidCategory As Guid, <[In](), MarshalAs(UnmanagedType.LPWStr)> ByVal szInstance As String, <[In]()> ByVal Filter As Guid) As Integer

    <PreserveSig()> _
    <Obsolete("This method has not been tested.", False)> _
    Shadows Function RegisterFilter(<[In]()> ByVal clsidFilter As Guid, <[In](), MarshalAs(UnmanagedType.LPWStr)> ByVal Name As String, <[In](), Out()> ByVal ppMoniker As IMoniker, <[In]()> ByVal pclsidCategory As DsGuid, <[In](), MarshalAs(UnmanagedType.LPWStr)> ByVal szInstance As String, <[In]()> ByVal prf2 As Object) As Integer

    ' GUID *
    ' GUID *
    <PreserveSig()> _
    Shadows Function EnumMatchingFilters(<Out()> ByRef ppEnum As IEnumMoniker, <[In]()> ByVal dwFlags As Integer, <[In](), MarshalAs(UnmanagedType.Bool)> ByVal bExactMatch As Boolean, <[In]()> ByVal dwMerit As Merit, <[In](), MarshalAs(UnmanagedType.Bool)> ByVal bInputNeeded As Boolean, <[In]()> ByVal cInputTypes As Integer, _
     <[In](), MarshalAs(UnmanagedType.LPArray, ArraySubType:=UnmanagedType.Struct)> ByVal pInputTypes As Guid(), <[In]()> ByVal pMedIn As RegPinMedium, <[In]()> ByVal pPinCategoryIn As DsGuid, <[In](), MarshalAs(UnmanagedType.Bool)> ByVal bRender As Boolean, <[In](), MarshalAs(UnmanagedType.Bool)> ByVal bOutputNeeded As Boolean, <[In]()> ByVal cOutputTypes As Integer, _
     <[In](), MarshalAs(UnmanagedType.LPArray, ArraySubType:=UnmanagedType.Struct)> ByVal pOutputTypes As Guid(), <[In]()> ByVal pMedOut As RegPinMedium, <[In]()> ByVal pPinCategoryOut As DsGuid) As Integer

    <PreserveSig()> _
    Function GetICreateDevEnum(<Out()> ByRef ppEnum As ICreateDevEnum) As Integer
End Interface

<ComImport(), System.Security.SuppressUnmanagedCodeSecurity(), Guid("436eee9c-264f-4242-90e1-4e330c107512"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)> _
Public Interface IMpeg2Demultiplexer
    <PreserveSig()> _
    Function CreateOutputPin(<[In]()> ByVal pMediaType As AMMediaType, <[In](), MarshalAs(UnmanagedType.LPWStr)> ByVal pszPinName As String, <Out()> ByRef ppIPin As IPin) As Integer

    <PreserveSig()> _
    Function SetOutputPinMediaType(<[In](), MarshalAs(UnmanagedType.LPWStr)> ByVal pszPinName As String, <[In]()> ByVal pMediaType As AMMediaType) As Integer

    <PreserveSig()> _
    Function DeleteOutputPin(<[In](), MarshalAs(UnmanagedType.LPWStr)> ByVal pszPinName As String) As Integer
End Interface

<ComImport(), System.Security.SuppressUnmanagedCodeSecurity(), Guid("B5730A90-1A2C-11cf-8C23-00AA006B6814"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)> _
Public Interface IAMExtDevice
    <PreserveSig()> _
    Function GetCapability(<[In]()> ByVal Capability As ExtDeviceCaps, <Out()> ByRef pValue As ExtDeviceCaps, <Out()> ByRef pdblValue As Double) As Integer

    <PreserveSig()> _
    Function get_ExternalDeviceID(<Out(), MarshalAs(UnmanagedType.LPWStr)> ByRef ppszData As String) As Integer

    <PreserveSig()> _
    Function get_ExternalDeviceVersion(<Out(), MarshalAs(UnmanagedType.LPWStr)> ByRef ppszData As String) As Integer

    <PreserveSig()> _
    Function put_DevicePower(<[In]()> ByVal PowerMode As ExtDeviceCaps) As Integer

    <PreserveSig()> _
    Function get_DevicePower(<Out()> ByRef pPowerMode As ExtDeviceCaps) As Integer

    ' HEVENT
    'Active / Inactive
    <PreserveSig()> _
    Function Calibrate(<[In]()> ByVal hEvent As IntPtr, <[In]()> ByVal Mode As ExtTransportEdit, <Out()> ByRef pStatus As Integer) As Integer

    <PreserveSig()> _
    Function put_DevicePort(<[In]()> ByVal DevicePort As ExtDevicePort) As Integer

    <PreserveSig()> _
    Function get_DevicePort(<Out()> ByRef pDevicePort As ExtDevicePort) As Integer
End Interface

<ComImport(), System.Security.SuppressUnmanagedCodeSecurity(), Guid("D0E04C47-25B8-4369-925A-362A01D95444"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)> _
Public Interface IMPEG2StreamIdMap
    <PreserveSig()> _
    Function MapStreamId(<[In]()> ByVal ulStreamId As Integer, <[In]()> ByVal MediaSampleContent As MPEG2Program, <[In]()> ByVal ulSubstreamFilterValue As Integer, <[In]()> ByVal iDataOffset As Integer) As Integer

    <PreserveSig()> _
    Function UnmapStreamId(<[In]()> ByVal culStreamId As Integer, <[In](), MarshalAs(UnmanagedType.LPArray)> ByVal pulStreamId As Integer()) As Integer

#If ALLOW_UNTESTED_INTERFACES Then
    <PreserveSig(), Obsolete("Because of bug in DS 9.0c, you can't get the StreamId map from .NET", False)> _
    Function EnumStreamIdMap(<Out()> ByRef ppIEnumStreamIdMap As IEnumStreamIdMap) As Integer
#Else
	Function EnumStreamIdMap(<Out> ByRef ppIEnumStreamIdMap As Object) As Integer
#End If
End Interface

<ComImport(), System.Security.SuppressUnmanagedCodeSecurity(), Guid("A03CD5F0-3045-11cf-8C44-00AA006B6814"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)> _
Public Interface IAMExtTransport
    <PreserveSig()> _
    Function GetCapability(<[In]()> ByVal Capability As ExtTransportCaps, <Out()> ByRef pValue As Integer, <Out()> ByRef pdblValue As Double) As Integer

    <PreserveSig()> _
    Function put_MediaState(<[In]()> ByVal State As ExtTransportMediaStates) As Integer

    <PreserveSig()> _
    Function get_MediaState(<Out()> ByRef pState As ExtTransportMediaStates) As Integer

    <PreserveSig()> _
    Function put_LocalControl(<[In]()> ByVal State As Integer) As Integer

    <PreserveSig()> _
    Function get_LocalControl(<Out()> ByRef pState As Integer) As Integer

    <PreserveSig()> _
    Function GetStatus(<[In]()> ByVal StatusItem As ExtTransportStatus, <Out()> ByRef pValue As Integer) As Integer

    <PreserveSig()> _
    Function GetTransportBasicParameters(<[In]()> ByVal Param As ExtTransportParameters, <Out()> ByRef pValue As Integer, <Out(), MarshalAs(UnmanagedType.LPWStr)> ByRef ppszData As String) As Integer

    <PreserveSig()> _
    Function SetTransportBasicParameters(<[In]()> ByVal Param As ExtTransportParameters, <[In]()> ByVal Value As Integer, <[In](), MarshalAs(UnmanagedType.LPWStr)> ByVal pszData As String) As Integer

    <PreserveSig()> _
    Function GetTransportVideoParameters(<[In]()> ByVal Param As ExtTransportParameters, <Out()> ByRef pValue As Integer) As Integer

    <PreserveSig()> _
    Function SetTransportVideoParameters(<[In]()> ByVal Param As ExtTransportParameters, <[In]()> ByVal Value As Integer) As Integer

    <PreserveSig()> _
    Function GetTransportAudioParameters(<[In]()> ByVal Param As ExtTransportParameters, <Out()> ByRef pValue As Integer) As Integer

    <PreserveSig()> _
    Function SetTransportAudioParameters(<[In]()> ByVal Param As ExtTransportParameters, <[In]()> ByVal Value As ExtTransportAudio) As Integer

    <PreserveSig()> _
    Function put_Mode(<[In]()> ByVal Mode As ExtTransportModes) As Integer

    <PreserveSig()> _
    Function get_Mode(<Out()> ByRef pMode As ExtTransportModes) As Integer

    <PreserveSig()> _
    Function put_Rate(<[In]()> ByVal dblRate As Double) As Integer

    <PreserveSig()> _
    Function get_Rate(<Out()> ByRef pdblRate As Double) As Integer

    ' HEVENT
    <PreserveSig()> _
    Function GetChase(<Out()> ByRef pEnabled As Integer, <Out()> ByRef pOffset As Integer, <Out()> ByRef phEvent As IntPtr) As Integer

    ' HEVENT
    <PreserveSig()> _
    Function SetChase(<[In]()> ByVal Enable As Integer, <[In]()> ByVal Offset As Integer, <[In]()> ByVal hEvent As IntPtr) As Integer

    <PreserveSig()> _
    Function GetBump(<Out()> ByRef pSpeed As Integer, <Out()> ByRef pDuration As Integer) As Integer

    <PreserveSig()> _
    Function SetBump(<[In]()> ByVal Speed As Integer, <[In]()> ByVal Duration As Integer) As Integer

    <PreserveSig()> _
    Function get_AntiClogControl(<Out()> ByRef pEnabled As Integer) As Integer

    <PreserveSig()> _
    Function put_AntiClogControl(<[In]()> ByVal Enable As Integer) As Integer

    <PreserveSig()> _
    Function GetEditPropertySet(<[In]()> ByVal EditID As Integer, <Out()> ByRef pState As ExtTransportEdit) As Integer

    <PreserveSig()> _
    Function SetEditPropertySet(<[In](), Out()> ByRef pEditID As Integer, <[In]()> ByVal State As ExtTransportEdit) As Integer

    <PreserveSig()> _
    Function GetEditProperty(<[In]()> ByVal EditID As Integer, <[In]()> ByVal Param As ExtTransportEdit, <Out()> ByRef pValue As Integer) As Integer

    <PreserveSig()> _
    Function SetEditProperty(<[In]()> ByVal EditID As Integer, <[In]()> ByVal Param As ExtTransportEdit, <[In]()> ByVal Value As Integer) As Integer

    <PreserveSig()> _
    Function get_EditStart(<Out()> ByRef pValue As Integer) As Integer

    <PreserveSig()> _
    Function put_EditStart(<[In]()> ByVal Value As Integer) As Integer
End Interface

<ComImport(), System.Security.SuppressUnmanagedCodeSecurity(), Guid("4995f511-9ddb-4f12-bd3b-f04611807b79"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)> _
Public Interface IAMGraphBuilderCallback

    <PreserveSig()> _
    Function SelectedFilter(<[In]()> ByVal pMon As IMoniker) As Integer

    <PreserveSig()> _
    Function CreatedFilter(<[In]()> ByVal pFil As IBaseFilter) As Integer
End Interface

<ComImport(), System.Security.SuppressUnmanagedCodeSecurity(), Guid("56a868a5-0ad4-11ce-b03a-0020af0ba770"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)> _
Public Interface IQualityControl
    <PreserveSig()> _
    Function Notify(<[In]()> ByVal pSelf As IBaseFilter, <[In]()> ByVal q As Quality) As Integer

    <PreserveSig()> _
    Function SetSink(<[In]()> ByVal piqc As IQualityControl) As Integer
End Interface

<ComImport(), System.Security.SuppressUnmanagedCodeSecurity(), Guid("56ED71A0-AF5F-11D0-B3F0-00AA003761C5"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)> _
Public Interface IAMBufferNegotiation
    <PreserveSig()> _
    Function SuggestAllocatorProperties(<[In]()> ByVal pprop As AllocatorProperties) As Integer

    <PreserveSig()> _
    Function GetAllocatorProperties(<Out()> ByVal pprop As AllocatorProperties) As Integer
End Interface

<ComImport(), System.Security.SuppressUnmanagedCodeSecurity(), Guid("22320CB2-D41A-11d2-BF7C-D7CB9DF0BF93"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)> _
Public Interface IAMAudioRendererStats
    <PreserveSig()> _
    Function GetStatParam(<[In]()> ByVal dwParam As AMAudioRendererStatParam, <Out()> ByRef pdwParam1 As Integer, <Out()> ByRef pdwParam2 As Integer) As Integer
End Interface

<ComImport(), System.Security.SuppressUnmanagedCodeSecurity(), Guid("7B3A2F01-0751-48DD-B556-004785171C54"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)> _
Public Interface IRegisterServiceProvider
    <PreserveSig()> _
    Function RegisterService(<[In](), MarshalAs(UnmanagedType.LPStruct)> ByVal guidService As Guid, <[In](), MarshalAs(UnmanagedType.IUnknown)> ByVal pUnkObject As Object) As Integer
End Interface

<ComImport(), System.Security.SuppressUnmanagedCodeSecurity(), Guid("36b73883-c2c8-11cf-8b46-00805f6cef60"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)> _
Public Interface ISeekingPassThru
    Function Init(<[In](), MarshalAs(UnmanagedType.Bool)> ByVal bSupportRendering As Boolean, <[In]()> ByVal pPin As IPin) As Integer
End Interface

<ComImport(), System.Security.SuppressUnmanagedCodeSecurity(), Guid("4d5466b0-a49c-11d1-abe8-00a0c905f375"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)> _
Public Interface IAMClockAdjust
    <PreserveSig()> _
    Function SetClockDelta(<[In]()> ByVal rtDelta As Long) As Integer
End Interface

<ComImport(), System.Security.SuppressUnmanagedCodeSecurity(), Guid("56a868aa-0ad4-11ce-b03a-0020af0ba770"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)> _
Public Interface IAsyncReader
    <PreserveSig()> _
    Function RequestAllocator(<[In]()> ByVal pPreferred As IMemAllocator, <[In](), MarshalAs(UnmanagedType.LPStruct)> ByVal pProps As AllocatorProperties, <Out()> ByRef ppActual As IMemAllocator) As Integer

    <PreserveSig()> _
    Function Request(<[In]()> ByVal pSample As IMediaSample, <[In]()> ByVal dwUser As IntPtr) As Integer

    <PreserveSig()> _
    Function WaitForNext(<[In]()> ByVal dwTimeout As Integer, <Out()> ByRef ppSample As IMediaSample, <Out()> ByRef pdwUser As IntPtr) As Integer

    <PreserveSig()> _
    Function SyncReadAligned(<[In]()> ByVal pSample As IMediaSample) As Integer

    ' BYTE *
    <PreserveSig()> _
    Function SyncRead(<[In]()> ByVal llPosition As Long, <[In]()> ByVal lLength As Integer, <Out()> ByVal pBuffer As IntPtr) As Integer

    <PreserveSig()> _
    Function Length(<Out()> ByRef pTotal As Long, <Out()> ByRef pAvailable As Long) As Integer

    <PreserveSig()> _
    Function BeginFlush() As Integer

    <PreserveSig()> _
    Function EndFlush() As Integer
End Interface

<ComImport(), System.Security.SuppressUnmanagedCodeSecurity(), Guid("C6E13370-30AC-11d0-A18C-00A0C9118956"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)> _
Public Interface IAMCameraControl
    <PreserveSig()> _
    Function GetRange(<[In]()> ByVal [Property] As CameraControlProperty, <Out()> ByRef pMin As Integer, <Out()> ByRef pMax As Integer, <Out()> ByRef pSteppingDelta As Integer, <Out()> ByRef pDefault As Integer, <Out()> ByRef pCapsFlags As CameraControlFlags) As Integer

    <PreserveSig()> _
    Function [Set](<[In]()> ByVal [Property] As CameraControlProperty, <[In]()> ByVal lValue As Integer, <[In]()> ByVal Flags As CameraControlFlags) As Integer

    <PreserveSig()> _
    Function [Get](<[In]()> ByVal [Property] As CameraControlProperty, <Out()> ByRef lValue As Integer, <Out()> ByRef Flags As CameraControlFlags) As Integer
End Interface

<ComImport(), System.Security.SuppressUnmanagedCodeSecurity(), Guid("aaf38154-b80b-422f-91e6-b66467509a07"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)> _
Public Interface IFilterGraph3
    Inherits IFilterGraph2

    <PreserveSig()> _
    Shadows Function AddFilter(<[In]()> ByVal pFilter As IBaseFilter, <[In](), MarshalAs(UnmanagedType.LPWStr)> ByVal pName As String) As Integer

    <PreserveSig()> _
    Shadows Function RemoveFilter(<[In]()> ByVal pFilter As IBaseFilter) As Integer

    <PreserveSig()> _
    Shadows Function EnumFilters(<Out()> ByRef ppEnum As IEnumFilters) As Integer

    <PreserveSig()> _
    Shadows Function FindFilterByName(<[In](), MarshalAs(UnmanagedType.LPWStr)> ByVal pName As String, <Out()> ByRef ppFilter As IBaseFilter) As Integer

    <PreserveSig()> _
    Shadows Function ConnectDirect(<[In]()> ByVal ppinOut As IPin, <[In]()> ByVal ppinIn As IPin, <[In](), MarshalAs(UnmanagedType.LPStruct)> ByVal pmt As AMMediaType) As Integer

    <PreserveSig()> _
    Shadows Function Reconnect(<[In]()> ByVal ppin As IPin) As Integer

    <PreserveSig()> _
    Shadows Function Disconnect(<[In]()> ByVal ppin As IPin) As Integer

    <PreserveSig()> _
    Shadows Function SetDefaultSyncSource() As Integer

    <PreserveSig()> _
    Shadows Function Connect(<[In]()> ByVal ppinOut As IPin, <[In]()> ByVal ppinIn As IPin) As Integer

    <PreserveSig()> _
    Shadows Function Render(<[In]()> ByVal ppinOut As IPin) As Integer

    <PreserveSig()> _
    Shadows Function RenderFile(<[In](), MarshalAs(UnmanagedType.LPWStr)> ByVal lpcwstrFile As String, <[In](), MarshalAs(UnmanagedType.LPWStr)> ByVal lpcwstrPlayList As String) As Integer

    <PreserveSig()> _
    Shadows Function AddSourceFilter(<[In](), MarshalAs(UnmanagedType.LPWStr)> ByVal lpcwstrFileName As String, <[In](), MarshalAs(UnmanagedType.LPWStr)> ByVal lpcwstrFilterName As String, <Out()> ByRef ppFilter As IBaseFilter) As Integer

    <PreserveSig()> _
    Shadows Function SetLogFile(ByVal hFile As IntPtr) As Integer
    ' DWORD_PTR
    <PreserveSig()> _
    Shadows Function Abort() As Integer

    <PreserveSig()> _
    Shadows Function ShouldOperationContinue() As Integer

    <PreserveSig()> _
    Shadows Function AddSourceFilterForMoniker(<[In]()> ByVal pMoniker As IMoniker, <[In]()> ByVal pCtx As IBindCtx, <[In](), MarshalAs(UnmanagedType.LPWStr)> ByVal lpcwstrFilterName As String, <Out()> ByRef ppFilter As IBaseFilter) As Integer

    <PreserveSig()> _
    Shadows Function ReconnectEx(<[In]()> ByVal ppin As IPin, <[In]()> ByVal pmt As AMMediaType) As Integer

    ' DWORD *
    <PreserveSig()> _
    Shadows Function RenderEx(<[In]()> ByVal pPinOut As IPin, <[In]()> ByVal dwFlags As AMRenderExFlags, <[In]()> ByVal pvContext As IntPtr) As Integer

    <PreserveSig()> _
    Function SetSyncSourceEx(ByVal pClockForMostOfFilterGraph As IReferenceClock, ByVal pClockForFilter As IReferenceClock, ByVal pFilter As IBaseFilter) As Integer

End Interface

<ComImport(), System.Security.SuppressUnmanagedCodeSecurity(), Guid("0e26a181-f40c-4635-8786-976284b52981"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)> _
Public Interface IAMPluginControl
    <PreserveSig()> _
    Function GetPreferredClsid(<[In](), MarshalAs(UnmanagedType.LPStruct)> ByVal subType As Guid, ByRef clsid As Guid) As Integer

    <PreserveSig()> _
    Function GetPreferredClsidByIndex(ByVal index As Integer, ByRef subType As Guid, ByRef clsid As Guid) As Integer

    <PreserveSig()> _
    Function SetPreferredClsid(<[In](), MarshalAs(UnmanagedType.LPStruct)> ByVal subType As Guid, <[In](), MarshalAs(UnmanagedType.LPStruct)> ByVal clsid As DsGuid) As Integer

    <PreserveSig()> _
    Function IsDisabled(<[In](), MarshalAs(UnmanagedType.LPStruct)> ByVal clsid As Guid) As Integer

    <PreserveSig()> _
    Function GetDisabledByIndex(ByVal index As Integer, ByRef clsid As Guid) As Integer

    <PreserveSig()> _
    Function SetDisabled(<[In](), MarshalAs(UnmanagedType.LPStruct)> ByVal clsid As Guid, ByVal disabled As Boolean) As Integer

    <PreserveSig()> _
    Function IsLegacyDisabled(<MarshalAs(UnmanagedType.LPWStr)> ByVal dllName As String) As Integer
End Interface

<ComImport(), System.Security.SuppressUnmanagedCodeSecurity(), Guid("cf7b26fc-9a00-485b-8147-3e789d5e8f67"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)> _
Public Interface IAMAsyncReaderTimestampScaling
    <PreserveSig()> _
    Function GetTimestampMode(<MarshalAs(UnmanagedType.Bool)> ByRef pfRaw As Boolean) As Integer

    <PreserveSig()> _
    Function SetTimestampMode(<MarshalAs(UnmanagedType.Bool)> ByVal fRaw As Boolean) As Integer
End Interface

