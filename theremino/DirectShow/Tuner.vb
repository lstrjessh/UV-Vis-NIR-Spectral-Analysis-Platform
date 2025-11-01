
Imports System.Runtime.InteropServices
Imports System.Runtime.InteropServices.ComTypes

Namespace BDA

    <ComImport(), Guid("5FFDC5E6-B83A-4b55-B6E8-C69E765FE9DB")> _
    Public Class TuningSpace
    End Class

    <ComImport(), Guid("8A674B4C-1F63-11d3-B64C-00C04F79498E")> _
    Public Class AnalogRadioTuningSpace
    End Class

    <ComImport(), Guid("F9769A06-7ACA-4e39-9CFB-97BB35F0E77E")> _
    Public Class AuxInTuningSpace
    End Class

    <ComImport(), Guid("8A674B4D-1F63-11d3-B64C-00C04F79498E")> _
    Public Class AnalogTVTuningSpace
    End Class

    <ComImport(), Guid("1BE49F30-0E1B-11d3-9D8E-00C04F72D980")> _
    Public Class LanguageComponentType
    End Class

    <ComImport(), Guid("418008F3-CF67-4668-9628-10DC52BE1D08")> _
    Public Class MPEG2ComponentType
    End Class

    <ComImport(), Guid("A8DCF3D5-0780-4ef4-8A83-2CFFAACB8ACE")> _
    Public Class ATSCComponentType
    End Class

    <ComImport(), Guid("809B6661-94C4-49e6-B6EC-3F0F862215AA")> _
    Public Class Components
    End Class

    <ComImport(), Guid("59DC47A8-116C-11d3-9D8E-00C04F72D980")> _
    Public Class Component
    End Class

    <ComImport(), Guid("055CB2D7-2969-45cd-914B-76890722F112")> _
    Public Class MPEG2Component
    End Class

    <ComImport(), Guid("B46E0D38-AB35-4a06-A137-70576B01B39F")> _
    Public Class TuneRequest
    End Class

    <ComImport(), Guid("0369B4E5-45B6-11d3-B650-00C04F79498E")> _
    Public Class ChannelTuneRequest
    End Class

    <ComImport(), Guid("0369B4E6-45B6-11d3-B650-00C04F79498E")> _
    Public Class ATSCChannelTuneRequest
    End Class

    <ComImport(), Guid("0955AC62-BF2E-4cba-A2B9-A63F772D46CF")> _
    Public Class MPEG2TuneRequest
    End Class

    <ComImport(), Guid("0888C883-AC4F-4943-B516-2C38D9B34562")> _
    Public Class Locator
    End Class

    <ComImport(), Guid("8872FF1B-98FA-4d7a-8D93-C9F1055F85BB")> _
    Public Class ATSCLocator
    End Class

    <ComImport(), Guid("C531D9FD-9685-4028-8B68-6E1232079F1E")> _
    Public Class DVBCLocator
    End Class

    <ComImport(), Guid("15D6504A-5494-499c-886C-973C9E53B9F1")> _
    Public Class DVBTuneRequest
    End Class

    <ComImport(), Guid("A1A2B1C4-0E3A-11d3-9D8E-00C04F72D980")> _
    Public Class ComponentTypes
    End Class

    <ComImport(), Guid("823535A0-0318-11d3-9D8E-00C04F72D980")> _
    Public Class ComponentType
    End Class

    <ComImport(), Guid("A2E30750-6C3D-11d3-B653-00C04F79498E")> _
    Public Class ATSCTuningSpace
    End Class

    <ComImport(), Guid("C6B14B32-76AA-4a86-A7AC-5C79AAF58DA7")> _
    Public Class DVBTuningSpace
    End Class

    <ComImport(), Guid("B64016F3-C9A2-4066-96F0-BD9563314726")> _
    Public Class DVBSTuningSpace
    End Class

    <ComImport(), Guid("9CD64701-BDF3-4d14-8E03-F12983D86664")> _
    Public Class DVBTLocator
    End Class

    <ComImport(), Guid("1DF7D126-4050-47f0-A7CF-4C4CA9241333")> _
    Public Class DVBSLocator
    End Class

    <ComImport(), Guid("8A674B49-1F63-11d3-B64C-00C04F79498E")> _
    Public Class CreatePropBagOnRegKey
    End Class

    <ComImport(), Guid("D02AAC50-027E-11d3-9D8E-00C04F72D980")> _
    Public Class SystemTuningSpaces
    End Class

    <ComImport(), Guid("2C63E4EB-4CEA-41b8-919C-E947EA19A77C")> _
    Public Class MPEG2TuneRequestFactory
    End Class

    <ComImport(), Guid("CC829A2F-3365-463f-AF13-81DBB6F3A555")> _
    Public Class ChannelIDTuningSpace
    End Class

    <ComImport(), Guid("03C06416-D127-407A-AB4C-FDD279ABBE5D")> _
    Public Class DigitalCableLocator
    End Class

    <ComImport(), Guid("E77026B0-B97F-4cbb-B7FB-F4F03AD69F11")> _
    Public Class PersistTuneXmlUtility
    End Class

    <ComImport(), Guid("26EC0B63-AA90-458A-8DF4-5659F2C8A18A")> _
    Public Class DigitalCableTuneRequest
    End Class

    <ComImport(), Guid("D9BB4CEE-B87A-47F1-AC92-B08D9C7813FC")> _
    Public Class DigitalCableTuningSpace
    End Class

    <ComImport(), Guid("28AB0005-E845-4FFA-AA9B-F4665236141C")> _
    Public Class AnalogAudioComponentType
    End Class

    <ComImport(), Guid("49638B91-48AB-48B7-A47A-7D0E75A08EDE")> _
    Public Class AnalogLocator
    End Class

#If ALLOW_UNTESTED_INTERFACES Then

    <ComImport(), Guid("6E50CC0D-C19B-4BF6-810B-5BD60761F5CC")> _
    Public Class DigitalLocator
    End Class

    <ComImport(), Guid("3A9428A7-31A4-45e9-9EFB-E055BF7BB3DB")> _
    Public Class ChannelIDTuneRequest
    End Class

    <ComImport(), Guid("6504AFED-A629-455c-A7F1-04964DEA5CC4")> _
    Public Class ISDBSLocator
    End Class

    <ComImport(), Guid("6438570B-0C08-4a25-9504-8012BB4D50CF")> _
    Public Class TunerMarshaler
    End Class

    <ComImport(), Guid("C20447FC-EC60-475e-813F-D2B0A6DECEFE")> _
    Public Class ESEventService
    End Class

    <ComImport(), Guid("8E8A07DA-71F8-40c1-A929-5E3A868AC2C6")> _
    Public Class ESEventFactory
    End Class

    Public Enum TunerLockType
        None = &H0
        WithinScanSensingRange = &H1
        Locked = &H2
    End Enum

    Public Enum LNB_Source
        NOT_SET = -1
        NOT_DEFINED = 0
        A = 1
        B = 2
        C = 3
        D = 4
        MAX
    End Enum

#End If


#If ALLOW_UNTESTED_INTERFACES Then

    <ComImport(), System.Security.SuppressUnmanagedCodeSecurity(), Guid("BAD7753B-6B37-4810-AE57-3CE0C4A9E6CB"), InterfaceType(ComInterfaceType.InterfaceIsDual)> _
    Public Interface IDigitalCableTuneRequest
        Inherits IATSCChannelTuneRequest

        <PreserveSig()> _
        Shadows Function get_TuningSpace(<Out()> ByRef TuningSpace As ITuningSpace) As Integer

        <PreserveSig()> _
        Shadows Function get_Components(<Out()> ByRef Components As IComponents) As Integer

        <PreserveSig()> _
        Shadows Function Clone(<Out()> ByRef NewTuneRequest As ITuneRequest) As Integer

        <PreserveSig()> _
        Shadows Function get_Locator(<Out()> ByRef Locator As ILocator) As Integer

        <PreserveSig()> _
        Shadows Function put_Locator(<[In]()> ByVal Locator As ILocator) As Integer

        <PreserveSig()> _
        Shadows Function get_Channel(<Out()> ByRef Channel As Integer) As Integer

        <PreserveSig()> _
        Shadows Function put_Channel(<[In]()> ByVal Channel As Integer) As Integer

        <PreserveSig()> _
        Shadows Function get_MinorChannel(<Out()> ByRef MinorChannel As Integer) As Integer

        <PreserveSig()> _
        Shadows Function put_MinorChannel(<[In]()> ByVal MinorChannel As Integer) As Integer

        <PreserveSig()> _
        Function get_MajorChannel(<Out()> ByRef pMajorChannel As Integer) As Integer

        <PreserveSig()> _
        Function put_MajorChannel(<[In]()> ByVal MajorChannel As Integer) As Integer

        <PreserveSig()> _
        Function get_SourceID(<Out()> ByRef pSourceID As Integer) As Integer

        <PreserveSig()> _
        Function put_SourceID(<[In]()> ByVal SourceID As Integer) As Integer
    End Interface

    <ComImport(), System.Security.SuppressUnmanagedCodeSecurity(), Guid("04BBD195-0E2D-4593-9BD5-4F908BC33CF5"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)> _
    Public Interface IScanningTunerEx
        Inherits IScanningTuner

        <PreserveSig()> _
        Shadows Function get_TuningSpace(<Out()> ByRef TuningSpace As ITuningSpace) As Integer

        <PreserveSig()> _
        Shadows Function put_TuningSpace(<[In]()> ByVal TuningSpace As ITuningSpace) As Integer

        <PreserveSig()> _
        Shadows Function EnumTuningSpaces(<Out()> ByRef ppEnum As IEnumTuningSpaces) As Integer

        <PreserveSig()> _
        Shadows Function get_TuneRequest(<Out()> ByRef TuneRequest As ITuneRequest) As Integer

        <PreserveSig()> _
        Shadows Function put_TuneRequest(<[In]()> ByVal TuneRequest As ITuneRequest) As Integer

        <PreserveSig()> _
        Shadows Function Validate(<[In]()> ByVal TuneRequest As ITuneRequest) As Integer

        <PreserveSig()> _
        Shadows Function get_PreferredComponentTypes(<Out()> ByRef ComponentTypes As IComponentTypes) As Integer

        <PreserveSig()> _
        Shadows Function put_PreferredComponentTypes(<[In]()> ByVal ComponentTypes As IComponentTypes) As Integer

        <PreserveSig()> _
        Shadows Function get_SignalStrength(<Out()> ByRef Strength As Integer) As Integer

        <PreserveSig()> _
        Shadows Function TriggerSignalEvents(<[In]()> ByVal Interval As Integer) As Integer

        <PreserveSig()> _
        Shadows Function SeekUp() As Integer

        <PreserveSig()> _
        Shadows Function SeekDown() As Integer

        <PreserveSig()> _
        Shadows Function ScanUp(<[In]()> ByVal MillisecondsPause As Integer) As Integer

        <PreserveSig()> _
        Shadows Function ScanDown(<Out()> ByRef MillisecondsPause As Integer) As Integer

        <PreserveSig()> _
        Shadows Function AutoProgram() As Integer

        <PreserveSig()> _
        Function GetCurrentLocator(<Out()> ByRef pILocator As ILocator) As Integer

        <PreserveSig()> _
        Function PerformExhaustiveScan(<[In]()> ByVal dwLowerFreq As Integer, <[In]()> ByVal dwHigherFreq As Integer, <[In](), MarshalAs(UnmanagedType.VariantBool)> ByVal bFineTune As Boolean, <[In]()> ByVal hEvent As IntPtr) As Integer

        <PreserveSig()> _
        Function TerminateCurrentScan(<Out()> ByRef pcurrentFreq As Integer) As Integer

        <PreserveSig()> _
        Function ResumeCurrentScan(<[In]()> ByVal hEvent As IntPtr) As Integer

        <PreserveSig()> _
        Function GetTunerScanningCapability(<Out(), MarshalAs(UnmanagedType.Bool)> ByRef HardwareAssistedScanning As Boolean, <[In](), Out()> ByRef NumStandardsSupported As Integer, <[In](), Out()> ByVal BroadcastStandards As Guid()) As Integer

        <PreserveSig()> _
        Function GetTunerStatus(<Out()> ByRef SecondsLeft As Integer, <Out()> ByRef CurrentLockType As TunerLockType, <Out(), MarshalAs(UnmanagedType.Bool)> ByRef AutoDetect As Boolean, <Out()> ByRef CurrentFreq As Integer) As Integer

        <PreserveSig()> _
        Function GetCurrentTunerStandardCapability(<[In]()> ByVal CurrentBroadcastStandard As Guid, <Out()> ByRef SettlingTime As Integer, <Out()> ByRef TvStandardsSupported As AnalogVideoStandard) As Integer

        <PreserveSig()> _
        Function SetScanSignalTypeFilter(<[In]()> ByVal ScanModulationTypes As ScanModulationTypes, <[In]()> ByVal AnalogVideoStandard As AnalogVideoStandard) As Integer
    End Interface

    <ComImport(), System.Security.SuppressUnmanagedCodeSecurity(), Guid("359B3901-572C-4854-BB49-CDEF66606A25"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)> _
    Public Interface IRegisterTuner
        Function Register(ByVal pTuner As ITuner, ByVal pGraph As IGraphBuilder) As Integer

        Function Unregister() As Integer
    End Interface

    <ComImport(), System.Security.SuppressUnmanagedCodeSecurity(), Guid("1DFD0A5C-0284-11d3-9D8E-00C04F72D980"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)> _
    Public Interface IScanningTuner
        Inherits ITuner

        <PreserveSig()> _
        Shadows Function get_TuningSpace(<Out()> ByRef TuningSpace As ITuningSpace) As Integer

        <PreserveSig()> _
        Shadows Function put_TuningSpace(<[In]()> ByVal TuningSpace As ITuningSpace) As Integer

        <PreserveSig()> _
        Shadows Function EnumTuningSpaces(<Out()> ByRef ppEnum As IEnumTuningSpaces) As Integer

        <PreserveSig()> _
        Shadows Function get_TuneRequest(<Out()> ByRef TuneRequest As ITuneRequest) As Integer

        <PreserveSig()> _
        Shadows Function put_TuneRequest(<[In]()> ByVal TuneRequest As ITuneRequest) As Integer

        <PreserveSig()> _
        Shadows Function Validate(<[In]()> ByVal TuneRequest As ITuneRequest) As Integer

        <PreserveSig()> _
        Shadows Function get_PreferredComponentTypes(<Out()> ByRef ComponentTypes As IComponentTypes) As Integer

        <PreserveSig()> _
        Shadows Function put_PreferredComponentTypes(<[In]()> ByVal ComponentTypes As IComponentTypes) As Integer

        <PreserveSig()> _
        Shadows Function get_SignalStrength(<Out()> ByRef Strength As Integer) As Integer

        <PreserveSig()> _
        Shadows Function TriggerSignalEvents(<[In]()> ByVal Interval As Integer) As Integer

        <PreserveSig()> _
        Function SeekUp() As Integer

        <PreserveSig()> _
        Function SeekDown() As Integer

        <PreserveSig()> _
        Function ScanUp(<[In]()> ByVal MillisecondsPause As Integer) As Integer

        <PreserveSig()> _
        Function ScanDown(<Out()> ByRef MillisecondsPause As Integer) As Integer

        <PreserveSig()> _
        Function AutoProgram() As Integer
    End Interface

    <ComImport(), System.Security.SuppressUnmanagedCodeSecurity(), Guid("6044634A-1733-4F99-B982-5FB12AFCE4F0"), InterfaceType(ComInterfaceType.InterfaceIsDual)> _
    Public Interface IDVBSLocator2
        Inherits IDVBSLocator
        <DispId(1)> _
        Shadows Property CarrierFrequency() As Integer

        <DispId(2)> _
        Shadows Property InnerFEC() As FECMethod

        <DispId(3)> _
        Shadows Property InnerFECRate() As BinaryConvolutionCodeRate

        <DispId(4)> _
        Shadows Property OuterFEC() As FECMethod

        <DispId(5)> _
        Shadows Property OuterFECRate() As BinaryConvolutionCodeRate

        <DispId(6)> _
        Shadows Property Modulation() As ModulationType

        <DispId(7)> _
        Shadows Property SymbolRate() As Integer

        <PreserveSig(), DispId(8)> _
        Overloads Function Clone() As <MarshalAs(UnmanagedType.[Interface])> ILocator

        <DispId(&H191)> _
        Shadows Property SignalPolarisation() As Polarisation

        <DispId(&H192)> _
        Shadows Property WestPosition() As Boolean

        <DispId(&H193)> _
        Shadows Property OrbitalPosition() As Integer

        <DispId(&H194)> _
        Shadows Property Azimuth() As Integer

        <DispId(&H195)> _
        Shadows Property Elevation() As Integer

        <DispId(&H196)> _
        Property DiseqLNBSource() As LNB_Source

        <DispId(&H197)> _
        Property LocalOscillatorOverrideLow() As Integer

        <DispId(&H198)> _
        Property LocalOscillatorOverrideHigh() As Integer

        <DispId(&H199)> _
        Property LocalLNBSwitchOverride() As Integer

        <DispId(410)> _
        Property LocalSpectralInversionOverride() As SpectralInversion

        <DispId(&H19B)> _
        Property SignalRollOff() As RollOff

        <DispId(&H19C)> _
        Property SignalPilot() As Pilot

    End Interface

    <ComImport(), System.Security.SuppressUnmanagedCodeSecurity(), Guid("6B80E96F-55E2-45AA-B754-0C23C8E7D5C1"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)> _
    Public Interface IESCloseMmiEvent
        Inherits IESEvent

        <PreserveSig()> _
        Shadows Function GetEventId() As Integer

        <PreserveSig()> _
        Shadows Function GetEventType() As Guid

        <PreserveSig()> _
        Shadows Function SetCompletionStatus(<[In]()> ByVal dwResult As Integer) As Integer

        <PreserveSig()> _
        Shadows Function GetData() As <MarshalAs(UnmanagedType.SafeArray, SafeArraySubType:=VarEnum.VT_UI1)> Byte()

        <PreserveSig()> _
        Shadows Function GetStringData() As <MarshalAs(UnmanagedType.BStr)> String

        <PreserveSig()> _
        Function GetDialogNumber() As Integer

    End Interface

    <ComImport(), System.Security.SuppressUnmanagedCodeSecurity(), Guid("1F0E5357-AF43-44E6-8547-654C645145D2"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)> _
    Public Interface IESEvent
        <PreserveSig()> _
        Function GetEventId() As Integer

        <PreserveSig()> _
        Function GetEventType() As Guid

        <PreserveSig()> _
        Function SetCompletionStatus(<[In]()> ByVal dwResult As Integer) As Integer

        <PreserveSig()> _
        Function GetData() As <MarshalAs(UnmanagedType.SafeArray, SafeArraySubType:=VarEnum.VT_UI1)> Byte()

        <PreserveSig()> _
        Function GetStringData() As <MarshalAs(UnmanagedType.BStr)> String

    End Interface

    <ComImport(), System.Security.SuppressUnmanagedCodeSecurity(), Guid("506A09B8-7F86-4E04-AC05-3303BFE8FC49"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)> _
    Public Interface IESEventFactory
        <PreserveSig()> _
        Function CreateESEvent(<[In](), MarshalAs(UnmanagedType.IUnknown)> ByVal pServiceProvider As Object, <[In]()> ByVal dwEventId As Integer, <[In]()> ByVal guidEventType As Guid, <[In]()> ByVal dwEventDataLength As Integer, <[In]()> ByRef pEventData As Byte, <[In](), MarshalAs(UnmanagedType.BStr)> ByVal bstrBaseUrl As String, _
         <[In](), MarshalAs(UnmanagedType.IUnknown)> ByVal pInitContext As Object) As <MarshalAs(UnmanagedType.[Interface])> IESEvent

    End Interface

    <ComImport(), System.Security.SuppressUnmanagedCodeSecurity(), Guid("ABD414BF-CFE5-4E5E-AF5B-4B4E49C5BFEB"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)> _
    Public Interface IESEvents
        <PreserveSig()> _
        Function OnESEventReceived(<[In]()> ByVal guidEventType As Guid, <[In](), MarshalAs(UnmanagedType.[Interface])> ByVal pESEvent As IESEvent) As Integer

    End Interface

    <ComImport(), System.Security.SuppressUnmanagedCodeSecurity(), Guid("ED89A619-4C06-4B2F-99EB-C7669B13047C"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)> _
    Public Interface IESEventService
        <PreserveSig()> _
        Function FireESEvent(<MarshalAs(UnmanagedType.[Interface])> ByVal pESEvent As IESEvent) As Integer

    End Interface

    <ComImport(), System.Security.SuppressUnmanagedCodeSecurity(), Guid("33B9DAAE-9309-491D-A051-BCAD2A70CD66"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown), TypeLibType(CShort(&H90))> _
    Public Interface IESEventServiceConfiguration
        <PreserveSig()> _
        Function SetParent(<MarshalAs(UnmanagedType.[Interface])> ByVal pEventService As IESEventService) As Integer

        <PreserveSig()> _
        Function RemoveParent() As Integer

        <PreserveSig()> _
        Function SetOwner(<MarshalAs(UnmanagedType.[Interface])> ByVal pESEvents As IESEvents) As Integer

        <PreserveSig()> _
        Function RemoveOwner() As Integer

        <PreserveSig()> _
        Function SetGraph(<MarshalAs(UnmanagedType.[Interface])> ByVal pGraph As IFilterGraph) As Integer

        <PreserveSig()> _
        Function RemoveGraph(<MarshalAs(UnmanagedType.[Interface])> ByVal pGraph As IFilterGraph) As Integer

    End Interface

    <ComImport(), System.Security.SuppressUnmanagedCodeSecurity(), Guid("BA9EDCB6-4D36-4CFE-8C56-87A6B0CA48E1"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)> _
    Public Interface IESFileExpiryDateEvent
        Inherits IESEvent

        <PreserveSig()> _
        Shadows Function GetEventId() As Integer

        <PreserveSig()> _
        Shadows Function GetEventType() As Guid

        <PreserveSig()> _
        Shadows Function SetCompletionStatus(<[In]()> ByVal dwResult As Integer) As Integer

        <PreserveSig()> _
        Shadows Function GetData() As <MarshalAs(UnmanagedType.SafeArray, SafeArraySubType:=VarEnum.VT_UI1)> Byte()

        <PreserveSig()> _
        Shadows Function GetStringData() As <MarshalAs(UnmanagedType.BStr)> String

        <PreserveSig()> _
        Function GetTunerId() As Guid

        <PreserveSig()> _
        Function GetExpiryDate() As Long

        <PreserveSig()> _
        Function GetFinalExpiryDate() As Long

        <PreserveSig()> _
        Function GetMaxRenewalCount() As Integer

        <PreserveSig()> _
        Function IsEntitlementTokenPresent() As Integer

        <PreserveSig()> _
        Function DoesExpireAfterFirstUse() As Integer

    End Interface

    <ComImport(), System.Security.SuppressUnmanagedCodeSecurity(), Guid("2017CB03-DC0F-4C24-83CA-36307B2CD19F"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)> _
    Public Interface IESIsdbCasResponseEvent
        Inherits IESEvent

        <PreserveSig()> _
        Shadows Function GetEventId() As Integer

        <PreserveSig()> _
        Shadows Function GetEventType() As Guid

        <PreserveSig()> _
        Shadows Function SetCompletionStatus(<[In]()> ByVal dwResult As Integer) As Integer

        <PreserveSig()> _
        Shadows Function GetData() As <MarshalAs(UnmanagedType.SafeArray, SafeArraySubType:=VarEnum.VT_UI1)> Byte()

        <PreserveSig()> _
        Shadows Function GetStringData() As <MarshalAs(UnmanagedType.BStr)> String

        <PreserveSig()> _
        Function GetRequestId() As Integer

        <PreserveSig()> _
        Function GetStatus() As Integer

        <PreserveSig()> _
        Function GetDataLength() As Integer

        <PreserveSig()> _
        Function GetResponseData() As <MarshalAs(UnmanagedType.SafeArray, SafeArraySubType:=VarEnum.VT_UI1)> Byte()

    End Interface

    <ComImport(), System.Security.SuppressUnmanagedCodeSecurity(), Guid("D5A48EF5-A81B-4DF0-ACAA-5E35E7EA45D4"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)> _
    Public Interface IESLicenseRenewalResultEvent
        Inherits IESEvent

        <PreserveSig()> _
        Shadows Function GetEventId() As Integer

        <PreserveSig()> _
        Shadows Function GetEventType() As Guid

        <PreserveSig()> _
        Shadows Function SetCompletionStatus(<[In]()> ByVal dwResult As Integer) As Integer

        <PreserveSig()> _
        Shadows Function GetData() As <MarshalAs(UnmanagedType.SafeArray, SafeArraySubType:=VarEnum.VT_UI1)> Byte()

        <PreserveSig()> _
        Shadows Function GetStringData() As <MarshalAs(UnmanagedType.BStr)> String

        <PreserveSig()> _
        Function GetCallersId() As Integer

        <PreserveSig()> _
        Function GetFileName() As <MarshalAs(UnmanagedType.BStr)> String

        <PreserveSig()> _
        Function IsRenewalSuccessful() As Integer

        <PreserveSig()> _
        Function IsCheckEntitlementCallRequired() As Integer

        <PreserveSig()> _
        Function GetDescrambledStatus() As Integer

        <PreserveSig()> _
        Function GetRenewalResultCode() As Integer

        <PreserveSig()> _
        Function GetCASFailureCode() As Integer

        <PreserveSig()> _
        Function GetRenewalHResult() As <MarshalAs(UnmanagedType.[Error])> Integer

        <PreserveSig()> _
        Function GetEntitlementTokenLength() As Integer

        <PreserveSig()> _
        Function GetEntitlementToken() As <MarshalAs(UnmanagedType.SafeArray, SafeArraySubType:=VarEnum.VT_UI1)> Byte()

        <PreserveSig()> _
        Function GetExpiryDate() As Long

    End Interface

    <ComImport(), System.Security.SuppressUnmanagedCodeSecurity(), Guid("BA4B6526-1A35-4635-8B56-3EC612746A8C"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)> _
    Public Interface IESOpenMmiEvent
        Inherits IESEvent

        <PreserveSig()> _
        Shadows Function GetEventId() As Integer

        <PreserveSig()> _
        Shadows Function GetEventType() As Guid

        <PreserveSig()> _
        Shadows Function SetCompletionStatus(<[In]()> ByVal dwResult As Integer) As Integer

        <PreserveSig()> _
        Shadows Function GetData() As <MarshalAs(UnmanagedType.SafeArray, SafeArraySubType:=VarEnum.VT_UI1)> Byte()

        <PreserveSig()> _
        Shadows Function GetStringData() As <MarshalAs(UnmanagedType.BStr)> String

        <PreserveSig()> _
        Function GetDialogNumber(ByRef pDialogRequest As Integer) As Integer

        <PreserveSig()> _
        Function GetDialogType() As Guid

        <PreserveSig()> _
        Function GetDialogData() As <MarshalAs(UnmanagedType.SafeArray, SafeArraySubType:=VarEnum.VT_UI1)> Byte()

        <PreserveSig()> _
        Function GetDialogStringData(<MarshalAs(UnmanagedType.BStr)> ByRef pbstrBaseUrl As String) As <MarshalAs(UnmanagedType.BStr)> String

    End Interface

    <ComImport(), System.Security.SuppressUnmanagedCodeSecurity(), Guid("54C7A5E8-C3BB-4F51-AF14-E0E2C0E34C6D"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)> _
    Public Interface IESRequestTunerEvent
        Inherits IESEvent

        <PreserveSig()> _
        Shadows Function GetEventId() As Integer

        <PreserveSig()> _
        Shadows Function GetEventType() As Guid

        <PreserveSig()> _
        Shadows Function SetCompletionStatus(<[In]()> ByVal dwResult As Integer) As Integer

        <PreserveSig()> _
        Shadows Function GetData() As <MarshalAs(UnmanagedType.SafeArray, SafeArraySubType:=VarEnum.VT_UI1)> Byte()

        <PreserveSig()> _
        Shadows Function GetStringData() As <MarshalAs(UnmanagedType.BStr)> String

        <PreserveSig()> _
        Function GetPriority() As Byte

        <PreserveSig()> _
        Function GetReason() As Byte

        <PreserveSig()> _
        Function GetConsequences() As Byte

        <PreserveSig()> _
        Function GetEstimatedTime() As Integer

    End Interface

    <ComImport(), System.Security.SuppressUnmanagedCodeSecurity(), Guid("8A24C46E-BB63-4664-8602-5D9C718C146D"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)> _
    Public Interface IESValueUpdatedEvent
        Inherits IESEvent

        <PreserveSig()> _
        Shadows Function GetEventId() As Integer

        <PreserveSig()> _
        Shadows Function GetEventType() As Guid

        <PreserveSig()> _
        Shadows Function SetCompletionStatus(<[In]()> ByVal dwResult As Integer) As Integer

        <PreserveSig()> _
        Shadows Function GetData() As <MarshalAs(UnmanagedType.SafeArray, SafeArraySubType:=VarEnum.VT_UI1)> Byte()

        <PreserveSig()> _
        Shadows Function GetStringData() As <MarshalAs(UnmanagedType.BStr)> String

        <PreserveSig()> _
        Function GetValueNames() As <MarshalAs(UnmanagedType.SafeArray, SafeArraySubType:=VarEnum.VT_BSTR)> String()

    End Interface

    <ComImport(), System.Security.SuppressUnmanagedCodeSecurity(), Guid("907E0B5C-E42D-4F04-91F0-26F401F36907"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)> _
    Public Interface IGpnvsCommonBase
        <PreserveSig()> _
        Function GetValueUpdateName() As <MarshalAs(UnmanagedType.BStr)> String

    End Interface

    <ComImport(), System.Security.SuppressUnmanagedCodeSecurity(), Guid("C9897087-E29C-473F-9E4B-7072123DEA14"), InterfaceType(ComInterfaceType.InterfaceIsDual)> _
    Public Interface IISDBSLocator
        Inherits IDVBSLocator
        <DispId(1)> _
        Shadows Property CarrierFrequency() As Integer

        <DispId(2)> _
        Shadows Property InnerFEC() As FECMethod

        <DispId(3)> _
        Shadows Property InnerFECRate() As BinaryConvolutionCodeRate

        <DispId(4)> _
        Shadows Property OuterFEC() As FECMethod

        <DispId(5)> _
        Shadows Property OuterFECRate() As BinaryConvolutionCodeRate

        <DispId(6)> _
        Shadows Property Modulation() As ModulationType

        <DispId(7)> _
        Shadows Property SymbolRate() As Integer

        <PreserveSig(), DispId(8)> _
        Overloads Function Clone() As <MarshalAs(UnmanagedType.[Interface])> ILocator

        <DispId(&H191)> _
        Shadows Property SignalPolarisation() As Polarisation

        <DispId(&H192)> _
        Shadows Property WestPosition() As Boolean

        <DispId(&H193)> _
        Shadows Property OrbitalPosition() As Integer

        <DispId(&H194)> _
        Shadows Property Azimuth() As Integer

        <DispId(&H195)> _
        Shadows Property Elevation() As Integer

    End Interface

    <ComImport(), System.Security.SuppressUnmanagedCodeSecurity(), Guid("ED3E0C66-18C8-4EA6-9300-F6841FDD35DC"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)> _
    Public Interface ITunerCapEx
        <DispId(&H60010000)> _
        ReadOnly Property Has608_708Caption() As Boolean

    End Interface

    <ComImport(), System.Security.SuppressUnmanagedCodeSecurity(), Guid("448A2EDF-AE95-4b43-A3CC-747843C453D4"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)> _
    Public Interface IDVBTLocator2
        Inherits IDVBTLocator

        <PreserveSig()> _
        Shadows Function get_CarrierFrequency(<Out()> ByRef Frequency As Integer) As Integer

        <PreserveSig()> _
        Shadows Function put_CarrierFrequency(<[In]()> ByVal Frequency As Integer) As Integer

        <PreserveSig()> _
        Shadows Function get_InnerFEC(<Out()> ByRef FEC As FECMethod) As Integer

        <PreserveSig()> _
        Shadows Function put_InnerFEC(<[In]()> ByVal FEC As FECMethod) As Integer

        <PreserveSig()> _
        Shadows Function get_InnerFECRate(<Out()> ByRef FEC As BinaryConvolutionCodeRate) As Integer

        <PreserveSig()> _
        Shadows Function put_InnerFECRate(<[In]()> ByVal FEC As BinaryConvolutionCodeRate) As Integer

        <PreserveSig()> _
        Shadows Function get_OuterFEC(<Out()> ByRef FEC As FECMethod) As Integer

        <PreserveSig()> _
        Shadows Function put_OuterFEC(<[In]()> ByVal FEC As FECMethod) As Integer

        <PreserveSig()> _
        Shadows Function get_OuterFECRate(<Out()> ByRef FEC As BinaryConvolutionCodeRate) As Integer

        <PreserveSig()> _
        Shadows Function put_OuterFECRate(<[In]()> ByVal FEC As BinaryConvolutionCodeRate) As Integer

        <PreserveSig()> _
        Shadows Function get_Modulation(<Out()> ByRef Modulation As ModulationType) As Integer

        <PreserveSig()> _
        Shadows Function put_Modulation(<[In]()> ByVal Modulation As ModulationType) As Integer

        <PreserveSig()> _
        Shadows Function get_SymbolRate(<Out()> ByRef Rate As Integer) As Integer

        <PreserveSig()> _
        Shadows Function put_SymbolRate(<[In]()> ByVal Rate As Integer) As Integer

        <PreserveSig()> _
        Shadows Function Clone(<Out()> ByRef NewLocator As ILocator) As Integer

        <PreserveSig()> _
        Shadows Function get_Bandwidth(<Out()> ByRef BandwidthVal As Integer) As Integer

        <PreserveSig()> _
        Shadows Function put_Bandwidth(<[In]()> ByVal BandwidthVal As Integer) As Integer

        <PreserveSig()> _
        Shadows Function get_LPInnerFEC(<Out()> ByRef FEC As FECMethod) As Integer

        <PreserveSig()> _
        Shadows Function put_LPInnerFEC(<[In]()> ByVal FEC As FECMethod) As Integer

        <PreserveSig()> _
        Shadows Function get_LPInnerFECRate(<Out()> ByRef FEC As BinaryConvolutionCodeRate) As Integer

        <PreserveSig()> _
        Shadows Function put_LPInnerFECRate(<[In]()> ByVal FEC As BinaryConvolutionCodeRate) As Integer

        <PreserveSig()> _
        Shadows Function get_HAlpha(<Out()> ByRef Alpha As HierarchyAlpha) As Integer

        <PreserveSig()> _
        Shadows Function put_HAlpha(<[In]()> ByVal Alpha As HierarchyAlpha) As Integer

        <PreserveSig()> _
        Shadows Function get_Guard(<Out()> ByRef GI As GuardInterval) As Integer

        <PreserveSig()> _
        Shadows Function put_Guard(<[In]()> ByVal GI As GuardInterval) As Integer

        <PreserveSig()> _
        Shadows Function get_Mode(<Out()> ByRef mode As TransmissionMode) As Integer

        <PreserveSig()> _
        Shadows Function put_Mode(<[In]()> ByVal mode As TransmissionMode) As Integer

        <PreserveSig()> _
        Shadows Function get_OtherFrequencyInUse(<Out(), MarshalAs(UnmanagedType.VariantBool)> ByRef OtherFrequencyInUseVal As Boolean) As Integer

        <PreserveSig()> _
        Shadows Function put_OtherFrequencyInUse(<[In](), MarshalAs(UnmanagedType.VariantBool)> ByVal OtherFrequencyInUseVal As Boolean) As Integer

        <PreserveSig()> _
        Function get_PhysicalLayerPipeId(ByRef PhysicalLayerPipeIdVal As Integer) As Integer

        <PreserveSig()> _
        Function put_PhysicalLayerPipeId(ByVal PhysicalLayerPipeIdVal As Integer) As Integer
    End Interface

#End If

    <ComImport(), System.Security.SuppressUnmanagedCodeSecurity(), Guid("061C6E30-E622-11d2-9493-00C04F72D980"), InterfaceType(ComInterfaceType.InterfaceIsDual)> _
    Public Interface ITuningSpace
        <PreserveSig()> _
        Function get_UniqueName(<Out(), MarshalAs(UnmanagedType.BStr)> ByRef Name As String) As Integer

        <PreserveSig()> _
        Function put_UniqueName(<[In](), MarshalAs(UnmanagedType.BStr)> ByVal Name As String) As Integer

        <PreserveSig()> _
        Function get_FriendlyName(<Out(), MarshalAs(UnmanagedType.BStr)> ByRef Name As String) As Integer

        <PreserveSig()> _
        Function put_FriendlyName(<[In](), MarshalAs(UnmanagedType.BStr)> ByVal Name As String) As Integer

        <PreserveSig()> _
        Function get_CLSID(<Out(), MarshalAs(UnmanagedType.BStr)> ByRef SpaceCLSID As String) As Integer

        <PreserveSig()> _
        Function get_NetworkType(<Out(), MarshalAs(UnmanagedType.BStr)> ByRef NetworkTypeGuid As String) As Integer

        <PreserveSig()> _
        Function put_NetworkType(<[In](), MarshalAs(UnmanagedType.BStr)> ByVal NetworkTypeGuid As String) As Integer

        <PreserveSig()> _
        Function get__NetworkType(<Out()> ByRef NetworkTypeGuid As Guid) As Integer

        <PreserveSig()> _
        Function put__NetworkType(<[In](), MarshalAs(UnmanagedType.LPStruct)> ByVal NetworkTypeGuid As Guid) As Integer

        <PreserveSig()> _
        Function CreateTuneRequest(<Out()> ByRef TuneRequest As ITuneRequest) As Integer

        ' IEnumGUID**
        <PreserveSig()> _
        Function EnumCategoryGUIDs(<Out(), MarshalAs(UnmanagedType.IUnknown)> ByRef ppEnum As Object) As Integer

        <PreserveSig()> _
        Function EnumDeviceMonikers(<Out()> ByRef ppEnum As IEnumMoniker) As Integer

        <PreserveSig()> _
        Function get_DefaultPreferredComponentTypes(<Out()> ByRef ComponentTypes As IComponentTypes) As Integer

        <PreserveSig()> _
        Function put_DefaultPreferredComponentTypes(<[In]()> ByVal NewComponentTypes As IComponentTypes) As Integer

        <PreserveSig()> _
        Function get_FrequencyMapping(<Out(), MarshalAs(UnmanagedType.BStr)> ByRef pMapping As String) As Integer

        <PreserveSig()> _
        Function put_FrequencyMapping(<[In](), MarshalAs(UnmanagedType.BStr)> ByVal Mapping As String) As Integer

        <PreserveSig()> _
        Function get_DefaultLocator(<Out()> ByRef LocatorVal As ILocator) As Integer

        <PreserveSig()> _
        Function put_DefaultLocator(<[In]()> ByVal LocatorVal As ILocator) As Integer

        <PreserveSig()> _
        Function Clone(<Out()> ByRef NewTS As ITuningSpace) As Integer
    End Interface

    <ComImport(), System.Security.SuppressUnmanagedCodeSecurity(), Guid("28C52640-018A-11d3-9D8E-00C04F72D980"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)> _
    Public Interface ITuner
        <PreserveSig()> _
        Function get_TuningSpace(<Out()> ByRef TuningSpace As ITuningSpace) As Integer

        <PreserveSig()> _
        Function put_TuningSpace(<[In]()> ByVal TuningSpace As ITuningSpace) As Integer

        <PreserveSig()> _
        Function EnumTuningSpaces(<Out()> ByRef ppEnum As IEnumTuningSpaces) As Integer

        <PreserveSig()> _
        Function get_TuneRequest(<Out()> ByRef TuneRequest As ITuneRequest) As Integer

        <PreserveSig()> _
        Function put_TuneRequest(<[In]()> ByVal TuneRequest As ITuneRequest) As Integer

        <PreserveSig()> _
        Function Validate(<[In]()> ByVal TuneRequest As ITuneRequest) As Integer

        <PreserveSig()> _
        Function get_PreferredComponentTypes(<Out()> ByRef ComponentTypes As IComponentTypes) As Integer

        <PreserveSig()> _
        Function put_PreferredComponentTypes(<[In]()> ByVal ComponentTypes As IComponentTypes) As Integer

        <PreserveSig()> _
        Function get_SignalStrength(<Out()> ByRef Strength As Integer) As Integer

        <PreserveSig()> _
        Function TriggerSignalEvents(<[In]()> ByVal Interval As Integer) As Integer
    End Interface

    <ComImport(), System.Security.SuppressUnmanagedCodeSecurity(), Guid("07DDC146-FC3D-11d2-9D8C-00C04F72D980"), InterfaceType(ComInterfaceType.InterfaceIsDual)> _
    Public Interface ITuneRequest
        <PreserveSig()> _
        Function get_TuningSpace(<Out()> ByRef TuningSpace As ITuningSpace) As Integer

        <PreserveSig()> _
        Function get_Components(<Out()> ByRef Components As IComponents) As Integer

        <PreserveSig()> _
        Function Clone(<Out()> ByRef NewTuneRequest As ITuneRequest) As Integer

        <PreserveSig()> _
        Function get_Locator(<Out()> ByRef Locator As ILocator) As Integer

        <PreserveSig()> _
        Function put_Locator(<[In]()> ByVal Locator As ILocator) As Integer
    End Interface

    <ComImport(), System.Security.SuppressUnmanagedCodeSecurity(), Guid("3D7C353C-0D04-45f1-A742-F97CC1188DC8"), InterfaceType(ComInterfaceType.InterfaceIsDual)> _
    Public Interface IDVBSLocator
        Inherits IDigitalLocator

        <PreserveSig()> _
        Shadows Function get_CarrierFrequency(<Out()> ByRef Frequency As Integer) As Integer

        <PreserveSig()> _
        Shadows Function put_CarrierFrequency(<[In]()> ByVal Frequency As Integer) As Integer

        <PreserveSig()> _
        Shadows Function get_InnerFEC(<Out()> ByRef FEC As FECMethod) As Integer

        <PreserveSig()> _
        Shadows Function put_InnerFEC(<[In]()> ByVal FEC As FECMethod) As Integer

        <PreserveSig()> _
        Shadows Function get_InnerFECRate(<Out()> ByRef FEC As BinaryConvolutionCodeRate) As Integer

        <PreserveSig()> _
        Shadows Function put_InnerFECRate(<[In]()> ByVal FEC As BinaryConvolutionCodeRate) As Integer

        <PreserveSig()> _
        Shadows Function get_OuterFEC(<Out()> ByRef FEC As FECMethod) As Integer

        <PreserveSig()> _
        Shadows Function put_OuterFEC(<[In]()> ByVal FEC As FECMethod) As Integer

        <PreserveSig()> _
        Shadows Function get_OuterFECRate(<Out()> ByRef FEC As BinaryConvolutionCodeRate) As Integer

        <PreserveSig()> _
        Shadows Function put_OuterFECRate(<[In]()> ByVal FEC As BinaryConvolutionCodeRate) As Integer

        <PreserveSig()> _
        Shadows Function get_Modulation(<Out()> ByRef Modulation As ModulationType) As Integer

        <PreserveSig()> _
        Shadows Function put_Modulation(<[In]()> ByVal Modulation As ModulationType) As Integer

        <PreserveSig()> _
        Shadows Function get_SymbolRate(<Out()> ByRef Rate As Integer) As Integer

        <PreserveSig()> _
        Shadows Function put_SymbolRate(<[In]()> ByVal Rate As Integer) As Integer

        <PreserveSig()> _
        Shadows Function Clone(<Out()> ByRef NewLocator As ILocator) As Integer

        <PreserveSig()> _
        Function get_SignalPolarisation(<Out()> ByRef PolarisationVal As Polarisation) As Integer

        <PreserveSig()> _
        Function put_SignalPolarisation(<[In]()> ByVal PolarisationVal As Polarisation) As Integer

        <PreserveSig()> _
        Function get_WestPosition(<Out(), MarshalAs(UnmanagedType.VariantBool)> ByRef WestLongitude As Boolean) As Integer

        <PreserveSig()> _
        Function put_WestPosition(<[In](), MarshalAs(UnmanagedType.VariantBool)> ByVal WestLongitude As Boolean) As Integer

        <PreserveSig()> _
        Function get_OrbitalPosition(<Out()> ByRef longitude As Integer) As Integer

        <PreserveSig()> _
        Function put_OrbitalPosition(<[In]()> ByVal longitude As Integer) As Integer

        <PreserveSig()> _
        Function get_Azimuth(<Out()> ByRef Azimuth As Integer) As Integer

        <PreserveSig()> _
        Function put_Azimuth(<[In]()> ByVal Azimuth As Integer) As Integer

        <PreserveSig()> _
        Function get_Elevation(<Out()> ByRef Elevation As Integer) As Integer

        <PreserveSig()> _
        Function put_Elevation(<[In]()> ByVal Elevation As Integer) As Integer
    End Interface

    <ComImport(), System.Security.SuppressUnmanagedCodeSecurity(), Guid("CDF7BE60-D954-42fd-A972-78971958E470"), InterfaceType(ComInterfaceType.InterfaceIsDual)> _
    Public Interface IDVBSTuningSpace
        Inherits IDVBTuningSpace2

        <PreserveSig()> _
        Shadows Function get_UniqueName(<Out(), MarshalAs(UnmanagedType.BStr)> ByRef Name As String) As Integer

        <PreserveSig()> _
        Shadows Function put_UniqueName(<[In](), MarshalAs(UnmanagedType.BStr)> ByVal Name As String) As Integer

        <PreserveSig()> _
        Shadows Function get_FriendlyName(<Out(), MarshalAs(UnmanagedType.BStr)> ByRef Name As String) As Integer

        <PreserveSig()> _
        Shadows Function put_FriendlyName(<[In](), MarshalAs(UnmanagedType.BStr)> ByVal Name As String) As Integer

        <PreserveSig()> _
        Shadows Function get_CLSID(<Out(), MarshalAs(UnmanagedType.BStr)> ByRef SpaceCLSID As String) As Integer

        <PreserveSig()> _
        Shadows Function get_NetworkType(<Out(), MarshalAs(UnmanagedType.BStr)> ByRef NetworkTypeGuid As String) As Integer

        <PreserveSig()> _
        Shadows Function put_NetworkType(<[In](), MarshalAs(UnmanagedType.BStr)> ByVal NetworkTypeGuid As String) As Integer

        <PreserveSig()> _
        Shadows Function get__NetworkType(<Out()> ByRef NetworkTypeGuid As Guid) As Integer

        <PreserveSig()> _
        Shadows Function put__NetworkType(<[In](), MarshalAs(UnmanagedType.LPStruct)> ByVal NetworkTypeGuid As Guid) As Integer

        <PreserveSig()> _
        Shadows Function CreateTuneRequest(<Out()> ByRef TuneRequest As ITuneRequest) As Integer

        ' IEnumGUID**
        <PreserveSig()> _
        Shadows Function EnumCategoryGUIDs(<Out(), MarshalAs(UnmanagedType.IUnknown)> ByRef ppEnum As Object) As Integer

        <PreserveSig()> _
        Shadows Function EnumDeviceMonikers(<Out()> ByRef ppEnum As IEnumMoniker) As Integer

        <PreserveSig()> _
        Shadows Function get_DefaultPreferredComponentTypes(<Out()> ByRef ComponentTypes As IComponentTypes) As Integer

        <PreserveSig()> _
        Shadows Function put_DefaultPreferredComponentTypes(<[In]()> ByVal NewComponentTypes As IComponentTypes) As Integer

        <PreserveSig()> _
        Shadows Function get_FrequencyMapping(<Out(), MarshalAs(UnmanagedType.BStr)> ByRef pMapping As String) As Integer

        <PreserveSig()> _
        Shadows Function put_FrequencyMapping(<[In](), MarshalAs(UnmanagedType.BStr)> ByVal Mapping As String) As Integer

        <PreserveSig()> _
        Shadows Function get_DefaultLocator(<Out()> ByRef LocatorVal As ILocator) As Integer

        <PreserveSig()> _
        Shadows Function put_DefaultLocator(<[In]()> ByVal LocatorVal As ILocator) As Integer

        <PreserveSig()> _
        Shadows Function Clone(<Out()> ByRef NewTS As ITuningSpace) As Integer

        <PreserveSig()> _
        Shadows Function get_SystemType(<Out()> ByRef SysType As DVBSystemType) As Integer

        <PreserveSig()> _
        Shadows Function put_SystemType(<[In]()> ByVal SysType As DVBSystemType) As Integer

        <PreserveSig()> _
        Shadows Function get_NetworkID(<Out()> ByRef NetworkID As Integer) As Integer

        <PreserveSig()> _
        Shadows Function put_NetworkID(<[In]()> ByVal NetworkID As Integer) As Integer

        <PreserveSig()> _
        Function get_LowOscillator(<Out()> ByRef LowOscillator As Integer) As Integer

        <PreserveSig()> _
        Function put_LowOscillator(<[In]()> ByVal LowOscillator As Integer) As Integer

        <PreserveSig()> _
        Function get_HighOscillator(<Out()> ByRef HighOscillator As Integer) As Integer

        <PreserveSig()> _
        Function put_HighOscillator(<[In]()> ByVal HighOscillator As Integer) As Integer

        <PreserveSig()> _
        Function get_LNBSwitch(<Out()> ByRef LNBSwitch As Integer) As Integer

        <PreserveSig()> _
        Function put_LNBSwitch(<[In]()> ByVal LNBSwitch As Integer) As Integer

        <PreserveSig()> _
        Function get_InputRange(<Out(), MarshalAs(UnmanagedType.BStr)> ByRef InputRange As String) As Integer

        <PreserveSig()> _
        Function put_InputRange(<Out(), MarshalAs(UnmanagedType.BStr)> ByVal InputRange As String) As Integer

        <PreserveSig()> _
        Function get_SpectralInversion(<Out()> ByRef SpectralInversionVal As SpectralInversion) As Integer

        <PreserveSig()> _
        Function put_SpectralInversion(<[In]()> ByVal SpectralInversionVal As SpectralInversion) As Integer
    End Interface

    <ComImport(), System.Security.SuppressUnmanagedCodeSecurity(), Guid("286D7F89-760C-4F89-80C4-66841D2507AA"), InterfaceType(ComInterfaceType.InterfaceIsDual)> _
    Public Interface ILocator
        <PreserveSig()> _
        Function get_CarrierFrequency(<Out()> ByRef Frequency As Integer) As Integer

        <PreserveSig()> _
        Function put_CarrierFrequency(<[In]()> ByVal Frequency As Integer) As Integer

        <PreserveSig()> _
        Function get_InnerFEC(<Out()> ByRef FEC As FECMethod) As Integer

        <PreserveSig()> _
        Function put_InnerFEC(<[In]()> ByVal FEC As FECMethod) As Integer

        <PreserveSig()> _
        Function get_InnerFECRate(<Out()> ByRef FEC As BinaryConvolutionCodeRate) As Integer

        <PreserveSig()> _
        Function put_InnerFECRate(<[In]()> ByVal FEC As BinaryConvolutionCodeRate) As Integer

        <PreserveSig()> _
        Function get_OuterFEC(<Out()> ByRef FEC As FECMethod) As Integer

        <PreserveSig()> _
        Function put_OuterFEC(<[In]()> ByVal FEC As FECMethod) As Integer

        <PreserveSig()> _
        Function get_OuterFECRate(<Out()> ByRef FEC As BinaryConvolutionCodeRate) As Integer

        <PreserveSig()> _
        Function put_OuterFECRate(<[In]()> ByVal FEC As BinaryConvolutionCodeRate) As Integer

        <PreserveSig()> _
        Function get_Modulation(<Out()> ByRef Modulation As ModulationType) As Integer

        <PreserveSig()> _
        Function put_Modulation(<[In]()> ByVal Modulation As ModulationType) As Integer

        <PreserveSig()> _
        Function get_SymbolRate(<Out()> ByRef Rate As Integer) As Integer

        <PreserveSig()> _
        Function put_SymbolRate(<[In]()> ByVal Rate As Integer) As Integer

        <PreserveSig()> _
        Function Clone(<Out()> ByRef NewLocator As ILocator) As Integer
    End Interface

    <ComImport(), System.Security.SuppressUnmanagedCodeSecurity(), Guid("ADA0B268-3B19-4e5b-ACC4-49F852BE13BA"), InterfaceType(ComInterfaceType.InterfaceIsDual)> _
    Public Interface IDVBTuningSpace
        Inherits ITuningSpace

        <PreserveSig()> _
        Shadows Function get_UniqueName(<Out(), MarshalAs(UnmanagedType.BStr)> ByRef Name As String) As Integer

        <PreserveSig()> _
        Shadows Function put_UniqueName(<[In](), MarshalAs(UnmanagedType.BStr)> ByVal Name As String) As Integer

        <PreserveSig()> _
        Shadows Function get_FriendlyName(<Out(), MarshalAs(UnmanagedType.BStr)> ByRef Name As String) As Integer

        <PreserveSig()> _
        Shadows Function put_FriendlyName(<[In](), MarshalAs(UnmanagedType.BStr)> ByVal Name As String) As Integer

        <PreserveSig()> _
        Shadows Function get_CLSID(<Out(), MarshalAs(UnmanagedType.BStr)> ByRef SpaceCLSID As String) As Integer

        <PreserveSig()> _
        Shadows Function get_NetworkType(<Out(), MarshalAs(UnmanagedType.BStr)> ByRef NetworkTypeGuid As String) As Integer

        <PreserveSig()> _
        Shadows Function put_NetworkType(<[In](), MarshalAs(UnmanagedType.BStr)> ByVal NetworkTypeGuid As String) As Integer

        <PreserveSig()> _
        Shadows Function get__NetworkType(<Out()> ByRef NetworkTypeGuid As Guid) As Integer

        <PreserveSig()> _
        Shadows Function put__NetworkType(<[In](), MarshalAs(UnmanagedType.LPStruct)> ByVal NetworkTypeGuid As Guid) As Integer

        <PreserveSig()> _
        Shadows Function CreateTuneRequest(<Out()> ByRef TuneRequest As ITuneRequest) As Integer

        ' IEnumGUID**
        <PreserveSig()> _
        Shadows Function EnumCategoryGUIDs(<Out(), MarshalAs(UnmanagedType.IUnknown)> ByRef ppEnum As Object) As Integer

        <PreserveSig()> _
        Shadows Function EnumDeviceMonikers(<Out()> ByRef ppEnum As IEnumMoniker) As Integer

        <PreserveSig()> _
        Shadows Function get_DefaultPreferredComponentTypes(<Out()> ByRef ComponentTypes As IComponentTypes) As Integer

        <PreserveSig()> _
        Shadows Function put_DefaultPreferredComponentTypes(<[In]()> ByVal NewComponentTypes As IComponentTypes) As Integer

        <PreserveSig()> _
        Shadows Function get_FrequencyMapping(<Out(), MarshalAs(UnmanagedType.BStr)> ByRef pMapping As String) As Integer

        <PreserveSig()> _
        Shadows Function put_FrequencyMapping(<[In](), MarshalAs(UnmanagedType.BStr)> ByVal Mapping As String) As Integer

        <PreserveSig()> _
        Shadows Function get_DefaultLocator(<Out()> ByRef LocatorVal As ILocator) As Integer

        <PreserveSig()> _
        Shadows Function put_DefaultLocator(<[In]()> ByVal LocatorVal As ILocator) As Integer

        <PreserveSig()> _
        Shadows Function Clone(<Out()> ByRef NewTS As ITuningSpace) As Integer

        <PreserveSig()> _
        Function get_SystemType(<Out()> ByRef SysType As DVBSystemType) As Integer

        <PreserveSig()> _
        Function put_SystemType(<[In]()> ByVal SysType As DVBSystemType) As Integer
    End Interface

    <ComImport(), System.Security.SuppressUnmanagedCodeSecurity(), Guid("843188B4-CE62-43db-966B-8145A094E040"), InterfaceType(ComInterfaceType.InterfaceIsDual)> _
    Public Interface IDVBTuningSpace2
        Inherits IDVBTuningSpace

        <PreserveSig()> _
        Shadows Function get_UniqueName(<Out(), MarshalAs(UnmanagedType.BStr)> ByRef Name As String) As Integer

        <PreserveSig()> _
        Shadows Function put_UniqueName(<[In](), MarshalAs(UnmanagedType.BStr)> ByVal Name As String) As Integer

        <PreserveSig()> _
        Shadows Function get_FriendlyName(<Out(), MarshalAs(UnmanagedType.BStr)> ByRef Name As String) As Integer

        <PreserveSig()> _
        Shadows Function put_FriendlyName(<[In](), MarshalAs(UnmanagedType.BStr)> ByVal Name As String) As Integer

        <PreserveSig()> _
        Shadows Function get_CLSID(<Out(), MarshalAs(UnmanagedType.BStr)> ByRef SpaceCLSID As String) As Integer

        <PreserveSig()> _
        Shadows Function get_NetworkType(<Out(), MarshalAs(UnmanagedType.BStr)> ByRef NetworkTypeGuid As String) As Integer

        <PreserveSig()> _
        Shadows Function put_NetworkType(<[In](), MarshalAs(UnmanagedType.BStr)> ByVal NetworkTypeGuid As String) As Integer

        <PreserveSig()> _
        Shadows Function get__NetworkType(<Out()> ByRef NetworkTypeGuid As Guid) As Integer

        <PreserveSig()> _
        Shadows Function put__NetworkType(<[In](), MarshalAs(UnmanagedType.LPStruct)> ByVal NetworkTypeGuid As Guid) As Integer

        <PreserveSig()> _
        Shadows Function CreateTuneRequest(<Out()> ByRef TuneRequest As ITuneRequest) As Integer

        ' IEnumGUID**
        <PreserveSig()> _
        Shadows Function EnumCategoryGUIDs(<Out(), MarshalAs(UnmanagedType.IUnknown)> ByRef ppEnum As Object) As Integer

        <PreserveSig()> _
        Shadows Function EnumDeviceMonikers(<Out()> ByRef ppEnum As IEnumMoniker) As Integer

        <PreserveSig()> _
        Shadows Function get_DefaultPreferredComponentTypes(<Out()> ByRef ComponentTypes As IComponentTypes) As Integer

        <PreserveSig()> _
        Shadows Function put_DefaultPreferredComponentTypes(<[In]()> ByVal NewComponentTypes As IComponentTypes) As Integer

        <PreserveSig()> _
        Shadows Function get_FrequencyMapping(<Out(), MarshalAs(UnmanagedType.BStr)> ByRef pMapping As String) As Integer

        <PreserveSig()> _
        Shadows Function put_FrequencyMapping(<[In](), MarshalAs(UnmanagedType.BStr)> ByVal Mapping As String) As Integer

        <PreserveSig()> _
        Shadows Function get_DefaultLocator(<Out()> ByRef LocatorVal As ILocator) As Integer

        <PreserveSig()> _
        Shadows Function put_DefaultLocator(<[In]()> ByVal LocatorVal As ILocator) As Integer

        <PreserveSig()> _
        Shadows Function Clone(<Out()> ByRef NewTS As ITuningSpace) As Integer

        <PreserveSig()> _
        Shadows Function get_SystemType(<Out()> ByRef SysType As DVBSystemType) As Integer

        <PreserveSig()> _
        Shadows Function put_SystemType(<[In]()> ByVal SysType As DVBSystemType) As Integer

        <PreserveSig()> _
        Function get_NetworkID(<Out()> ByRef NetworkID As Integer) As Integer

        <PreserveSig()> _
        Function put_NetworkID(<[In]()> ByVal NetworkID As Integer) As Integer
    End Interface

    <ComImport(), System.Security.SuppressUnmanagedCodeSecurity(), Guid("8B8EB248-FC2B-11d2-9D8C-00C04F72D980"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)> _
    Public Interface IEnumTuningSpaces
        Function [Next](<[In]()> ByVal celt As Integer, <Out(), MarshalAs(UnmanagedType.LPArray, SizeParamIndex:=0)> ByVal rgelt As ITuningSpace(), <[In]()> ByVal pceltFetched As IntPtr) As Integer

        Function Skip(<[In]()> ByVal celt As Integer) As Integer

        Function Reset() As Integer

        Function Clone(<Out()> ByRef ppEnum As IEnumTuningSpaces) As Integer
    End Interface

    ' because of _TuningSpacesForCLSID
    <ComImport(), System.Security.SuppressUnmanagedCodeSecurity(), CLSCompliant(False), Guid("5B692E84-E2F1-11d2-9493-00C04F72D980"), InterfaceType(ComInterfaceType.InterfaceIsDual)> _
    Public Interface ITuningSpaceContainer
        <PreserveSig()> _
        Function get_Count(<Out()> ByRef Count As Integer) As Integer

        <PreserveSig()> _
        Function get__NewEnum(<Out()> ByRef ppNewEnum As IEnumVARIANT) As Integer

        <PreserveSig()> _
        Function get_Item(<[In]()> ByVal varIndex As Object, <Out()> ByRef TuningSpace As ITuningSpace) As Integer

        <PreserveSig()> _
        Function put_Item(<[In]()> ByVal varIndex As Object, <[In]()> ByVal TuningSpace As ITuningSpace) As Integer

        <PreserveSig()> _
        Function TuningSpacesForCLSID(<[In](), MarshalAs(UnmanagedType.BStr)> ByVal SpaceCLSID As String, <Out()> ByRef NewColl As ITuningSpaces) As Integer

        <PreserveSig()> _
        Function _TuningSpacesForCLSID(<[In](), MarshalAs(UnmanagedType.LPStruct)> ByVal SpaceCLSID As Guid, <Out()> ByRef NewColl As ITuningSpaces) As Integer

        <PreserveSig()> _
        Function TuningSpacesForName(<[In](), MarshalAs(UnmanagedType.BStr)> ByVal Name As String, <Out()> ByRef NewColl As ITuningSpaces) As Integer

        <PreserveSig()> _
        Function FindID(<[In]()> ByVal TuningSpace As ITuningSpace, <Out()> ByRef ID As Integer) As Integer

        <PreserveSig()> _
        Function Add(<[In]()> ByVal TuningSpace As ITuningSpace, <Out()> ByRef NewIndex As Object) As Integer

        <PreserveSig()> _
        Function get_EnumTuningSpaces(<Out()> ByRef ppEnum As IEnumTuningSpaces) As Integer

        <PreserveSig()> _
        Function Remove(<[In]()> ByVal Index As Object) As Integer

        <PreserveSig()> _
        Function get_MaxCount(<Out()> ByRef MaxCount As Integer) As Integer

        <PreserveSig()> _
        Function put_MaxCount(<[In]()> ByVal MaxCount As Integer) As Integer
    End Interface

    <ComImport(), System.Security.SuppressUnmanagedCodeSecurity(), Guid("901284E4-33FE-4b69-8D63-634A596F3756"), InterfaceType(ComInterfaceType.InterfaceIsDual)> _
    Public Interface ITuningSpaces
        <PreserveSig()> _
        Function get_Count(<Out()> ByRef Count As Integer) As Integer

        <PreserveSig()> _
        Function get__NewEnum(<Out()> ByRef ppNewEnum As IEnumVARIANT) As Integer

        <PreserveSig()> _
        Function get_Item(<[In]()> ByVal varIndex As Object, <Out()> ByRef TuningSpace As ITuningSpace) As Integer

        <PreserveSig()> _
        Function get_EnumTuningSpaces(<Out()> ByRef NewEnum As IEnumTuningSpaces) As Integer
    End Interface

    <ComImport(), System.Security.SuppressUnmanagedCodeSecurity(), Guid("0D6F567E-A636-42bb-83BA-CE4C1704AFA2"), InterfaceType(ComInterfaceType.InterfaceIsDual)> _
    Public Interface IDVBTuneRequest
        Inherits ITuneRequest

        <PreserveSig()> _
        Shadows Function get_TuningSpace(<Out()> ByRef TuningSpace As ITuningSpace) As Integer

        <PreserveSig()> _
        Shadows Function get_Components(<Out()> ByRef Components As IComponents) As Integer

        <PreserveSig()> _
        Shadows Function Clone(<Out()> ByRef NewTuneRequest As ITuneRequest) As Integer

        <PreserveSig()> _
        Shadows Function get_Locator(<Out()> ByRef Locator As ILocator) As Integer

        <PreserveSig()> _
        Shadows Function put_Locator(<[In]()> ByVal Locator As ILocator) As Integer

        <PreserveSig()> _
        Function get_ONID(<Out()> ByRef ONID As Integer) As Integer

        <PreserveSig()> _
        Function put_ONID(<[In]()> ByVal ONID As Integer) As Integer

        <PreserveSig()> _
        Function get_TSID(<Out()> ByRef TSID As Integer) As Integer

        <PreserveSig()> _
        Function put_TSID(<[In]()> ByVal TSID As Integer) As Integer

        <PreserveSig()> _
        Function get_SID(<Out()> ByRef SID As Integer) As Integer

        <PreserveSig()> _
        Function put_SID(<[In]()> ByVal SID As Integer) As Integer
    End Interface

    <ComImport(), System.Security.SuppressUnmanagedCodeSecurity(), Guid("8664DA16-DDA2-42ac-926A-C18F9127C302"), InterfaceType(ComInterfaceType.InterfaceIsDual)> _
    Public Interface IDVBTLocator
        Inherits IDigitalLocator

        <PreserveSig()> _
        Shadows Function get_CarrierFrequency(<Out()> ByRef Frequency As Integer) As Integer

        <PreserveSig()> _
        Shadows Function put_CarrierFrequency(<[In]()> ByVal Frequency As Integer) As Integer

        <PreserveSig()> _
        Shadows Function get_InnerFEC(<Out()> ByRef FEC As FECMethod) As Integer

        <PreserveSig()> _
        Shadows Function put_InnerFEC(<[In]()> ByVal FEC As FECMethod) As Integer

        <PreserveSig()> _
        Shadows Function get_InnerFECRate(<Out()> ByRef FEC As BinaryConvolutionCodeRate) As Integer

        <PreserveSig()> _
        Shadows Function put_InnerFECRate(<[In]()> ByVal FEC As BinaryConvolutionCodeRate) As Integer

        <PreserveSig()> _
        Shadows Function get_OuterFEC(<Out()> ByRef FEC As FECMethod) As Integer

        <PreserveSig()> _
        Shadows Function put_OuterFEC(<[In]()> ByVal FEC As FECMethod) As Integer

        <PreserveSig()> _
        Shadows Function get_OuterFECRate(<Out()> ByRef FEC As BinaryConvolutionCodeRate) As Integer

        <PreserveSig()> _
        Shadows Function put_OuterFECRate(<[In]()> ByVal FEC As BinaryConvolutionCodeRate) As Integer

        <PreserveSig()> _
        Shadows Function get_Modulation(<Out()> ByRef Modulation As ModulationType) As Integer

        <PreserveSig()> _
        Shadows Function put_Modulation(<[In]()> ByVal Modulation As ModulationType) As Integer

        <PreserveSig()> _
        Shadows Function get_SymbolRate(<Out()> ByRef Rate As Integer) As Integer

        <PreserveSig()> _
        Shadows Function put_SymbolRate(<[In]()> ByVal Rate As Integer) As Integer

        <PreserveSig()> _
        Shadows Function Clone(<Out()> ByRef NewLocator As ILocator) As Integer

        <PreserveSig()> _
        Function get_Bandwidth(<Out()> ByRef BandwidthVal As Integer) As Integer

        <PreserveSig()> _
        Function put_Bandwidth(<[In]()> ByVal BandwidthVal As Integer) As Integer

        <PreserveSig()> _
        Function get_LPInnerFEC(<Out()> ByRef FEC As FECMethod) As Integer

        <PreserveSig()> _
        Function put_LPInnerFEC(<[In]()> ByVal FEC As FECMethod) As Integer

        <PreserveSig()> _
        Function get_LPInnerFECRate(<Out()> ByRef FEC As BinaryConvolutionCodeRate) As Integer

        <PreserveSig()> _
        Function put_LPInnerFECRate(<[In]()> ByVal FEC As BinaryConvolutionCodeRate) As Integer

        <PreserveSig()> _
        Function get_HAlpha(<Out()> ByRef Alpha As HierarchyAlpha) As Integer

        <PreserveSig()> _
        Function put_HAlpha(<[In]()> ByVal Alpha As HierarchyAlpha) As Integer

        <PreserveSig()> _
        Function get_Guard(<Out()> ByRef GI As GuardInterval) As Integer

        <PreserveSig()> _
        Function put_Guard(<[In]()> ByVal GI As GuardInterval) As Integer

        <PreserveSig()> _
        Function get_Mode(<Out()> ByRef mode As TransmissionMode) As Integer

        <PreserveSig()> _
        Function put_Mode(<[In]()> ByVal mode As TransmissionMode) As Integer

        <PreserveSig()> _
        Function get_OtherFrequencyInUse(<Out(), MarshalAs(UnmanagedType.VariantBool)> ByRef OtherFrequencyInUseVal As Boolean) As Integer

        <PreserveSig()> _
        Function put_OtherFrequencyInUse(<[In](), MarshalAs(UnmanagedType.VariantBool)> ByVal OtherFrequencyInUseVal As Boolean) As Integer
    End Interface

    <ComImport(), System.Security.SuppressUnmanagedCodeSecurity(), Guid("8A674B4A-1F63-11d3-B64C-00C04F79498E"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)> _
    Public Interface IEnumComponentTypes
        Function [Next](<[In]()> ByVal celt As Integer, <Out(), MarshalAs(UnmanagedType.LPArray, SizeParamIndex:=0)> ByVal rgelt As IComponentType(), <[In]()> ByVal pceltFetched As IntPtr) As Integer

        Function Skip(<[In]()> ByVal celt As Integer) As Integer

        Function Reset() As Integer

        Function Clone(<Out()> ByRef ppEnum As IEnumComponentTypes) As Integer
    End Interface

    <ComImport(), System.Security.SuppressUnmanagedCodeSecurity(), Guid("6A340DC0-0311-11d3-9D8E-00C04F72D980"), InterfaceType(ComInterfaceType.InterfaceIsDual)> _
    Public Interface IComponentType
        <PreserveSig()> _
        Function get_Category(<Out()> ByRef Category As ComponentCategory) As Integer

        <PreserveSig()> _
        Function put_Category(<[In]()> ByVal Category As ComponentCategory) As Integer

        <PreserveSig()> _
        Function get_MediaMajorType(<Out(), MarshalAs(UnmanagedType.BStr)> ByRef MediaMajorType As String) As Integer

        <PreserveSig()> _
        Function put_MediaMajorType(<[In](), MarshalAs(UnmanagedType.BStr)> ByVal MediaMajorType As String) As Integer

        <PreserveSig()> _
        Function get__MediaMajorType(<Out()> ByRef MediaMajorType As Guid) As Integer

        <PreserveSig()> _
        Function put__MediaMajorType(<[In](), MarshalAs(UnmanagedType.LPStruct)> ByVal MediaMajorType As Guid) As Integer

        <PreserveSig()> _
        Function get_MediaSubType(<Out(), MarshalAs(UnmanagedType.BStr)> ByRef MediaSubType As String) As Integer

        <PreserveSig()> _
        Function put_MediaSubType(<[In](), MarshalAs(UnmanagedType.BStr)> ByVal MediaSubType As String) As Integer

        <PreserveSig()> _
        Function get__MediaSubType(<Out()> ByRef MediaSubType As Guid) As Integer

        <PreserveSig()> _
        Function put__MediaSubType(<[In](), MarshalAs(UnmanagedType.LPStruct)> ByVal MediaSubType As Guid) As Integer

        <PreserveSig()> _
        Function get_MediaFormatType(<Out(), MarshalAs(UnmanagedType.BStr)> ByRef MediaFormatType As String) As Integer

        <PreserveSig()> _
        Function put_MediaFormatType(<[In](), MarshalAs(UnmanagedType.BStr)> ByVal MediaFormatType As String) As Integer

        <PreserveSig()> _
        Function get__MediaFormatType(<Out()> ByRef MediaFormatType As Guid) As Integer

        <PreserveSig()> _
        Function put__MediaFormatType(<[In](), MarshalAs(UnmanagedType.LPStruct)> ByVal MediaFormatType As Guid) As Integer

        <PreserveSig()> _
        Function get_MediaType(<Out()> ByVal MediaType As AMMediaType) As Integer

        <PreserveSig()> _
        Function put_MediaType(<[In]()> ByVal MediaType As AMMediaType) As Integer

        <PreserveSig()> _
        Function Clone(<Out()> ByRef NewCT As IComponentType) As Integer
    End Interface

    <ComImport(), System.Security.SuppressUnmanagedCodeSecurity(), Guid("0DC13D4A-0313-11d3-9D8E-00C04F72D980"), InterfaceType(ComInterfaceType.InterfaceIsDual)> _
    Public Interface IComponentTypes
        <PreserveSig()> _
        Function get_Count(<Out()> ByRef Count As Integer) As Integer

        <PreserveSig()> _
        Function get__NewEnum(<Out()> ByRef ppNewEnum As IEnumVARIANT) As Integer

        <PreserveSig()> _
        Function EnumComponentTypes(<Out()> ByRef ppNewEnum As IEnumComponentTypes) As Integer

        <PreserveSig()> _
        Function get_Item(<[In]()> ByVal varIndex As Object, <Out()> ByRef TuningSpace As IComponentType) As Integer

        <PreserveSig()> _
        Function put_Item(<[In]()> ByVal NewIndex As Object, <[In]()> ByVal ComponentType As IComponentType) As Integer

        <PreserveSig()> _
        Function Add(<[In]()> ByVal ComponentType As IComponentType, <Out()> ByRef NewIndex As Object) As Integer

        <PreserveSig()> _
        Function Remove(<[In]()> ByVal Index As Object) As Integer

        <PreserveSig()> _
        Function Clone(<Out()> ByRef NewList As IComponentTypes) As Integer
    End Interface

    <ComImport(), System.Security.SuppressUnmanagedCodeSecurity(), Guid("39A48091-FFFE-4182-A161-3FF802640E26"), InterfaceType(ComInterfaceType.InterfaceIsDual)> _
    Public Interface IComponentsNew
        <PreserveSig()> _
        Function get_Count(<Out()> ByRef Count As Integer) As Integer

        <PreserveSig()> _
        Function get__NewEnum(<Out()> ByRef ppNewEnum As IEnumVARIANT) As Integer

        <PreserveSig()> _
        Function EnumComponents(<Out()> ByRef ppNewEnum As IEnumComponents) As Integer

        <PreserveSig()> _
        Function get_Item(<[In]()> ByVal varIndex As Object, <Out()> ByRef TuningSpace As IComponent) As Integer

        <PreserveSig()> _
        Function Add(<[In]()> ByVal Component As IComponent, <Out()> ByRef NewIndex As Object) As Integer

        <PreserveSig()> _
        Function Remove(<[In]()> ByVal Index As Object) As Integer

        <PreserveSig()> _
        Function Clone(<Out()> ByRef NewList As IComponentsNew) As Integer

        <PreserveSig()> _
        Function put_Item(ByVal Index As Object, ByVal ppComponent As IComponent) As Integer
    End Interface

    <ComImport(), System.Security.SuppressUnmanagedCodeSecurity(), Guid("2A6E2939-2595-11d3-B64C-00C04F79498E"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)> _
    Public Interface IEnumComponents
        Function [Next](<[In]()> ByVal celt As Integer, <Out(), MarshalAs(UnmanagedType.LPArray, SizeParamIndex:=0)> ByVal rgelt As IComponent(), <[In]()> ByVal pceltFetched As IntPtr) As Integer

        Function Skip(<[In]()> ByVal celt As Integer) As Integer

        Function Reset() As Integer

        Function Clone(<Out()> ByRef ppEnum As IEnumComponents) As Integer
    End Interface

    <ComImport(), System.Security.SuppressUnmanagedCodeSecurity(), Guid("1A5576FC-0E19-11d3-9D8E-00C04F72D980"), InterfaceType(ComInterfaceType.InterfaceIsDual)> _
    Public Interface IComponent
        <PreserveSig()> _
        Function get_Type(<Out()> ByRef CT As IComponentType) As Integer

        <PreserveSig()> _
        Function put_Type(<[In]()> ByVal CT As IComponentType) As Integer

        <PreserveSig()> _
        Function get_DescLangID(<Out()> ByRef LangID As Integer) As Integer

        <PreserveSig()> _
        Function put_DescLangID(<[In]()> ByVal LangID As Integer) As Integer

        <PreserveSig()> _
        Function get_Status(<Out()> ByRef Status As ComponentStatus) As Integer

        <PreserveSig()> _
        Function put_Status(<[In]()> ByVal Status As ComponentStatus) As Integer

        <PreserveSig()> _
        Function get_Description(<Out(), MarshalAs(UnmanagedType.BStr)> ByRef Description As String) As Integer

        <PreserveSig()> _
        Function put_Description(<[In](), MarshalAs(UnmanagedType.BStr)> ByVal Description As String) As Integer

        <PreserveSig()> _
        Function Clone(<Out()> ByRef NewComponent As IComponent) As Integer
    End Interface

    <ComImport(), System.Security.SuppressUnmanagedCodeSecurity(), Guid("EB7D987F-8A01-42ad-B8AE-574DEEE44D1A"), InterfaceType(ComInterfaceType.InterfaceIsDual)> _
    Public Interface IMPEG2TuneRequest
        Inherits ITuneRequest

        <PreserveSig()> _
        Shadows Function get_TuningSpace(<Out()> ByRef TuningSpace As ITuningSpace) As Integer

        <PreserveSig()> _
        Shadows Function get_Components(<Out()> ByRef Components As IComponents) As Integer

        <PreserveSig()> _
        Shadows Function Clone(<Out()> ByRef NewTuneRequest As ITuneRequest) As Integer

        <PreserveSig()> _
        Shadows Function get_Locator(<Out()> ByRef Locator As ILocator) As Integer

        <PreserveSig()> _
        Shadows Function put_Locator(<[In]()> ByVal Locator As ILocator) As Integer

        <PreserveSig()> _
        Function get_TSID(<Out()> ByRef TSID As Integer) As Integer

        <PreserveSig()> _
        Function put_TSID(<[In]()> ByVal TSID As Integer) As Integer

        <PreserveSig()> _
        Function get_ProgNo(<Out()> ByRef ProgNo As Integer) As Integer

        <PreserveSig()> _
        Function put_ProgNo(<[In]()> ByVal ProgNo As Integer) As Integer
    End Interface

    <ComImport(), System.Security.SuppressUnmanagedCodeSecurity(), Guid("14E11ABD-EE37-4893-9EA1-6964DE933E39"), InterfaceType(ComInterfaceType.InterfaceIsDual)> _
    Public Interface IMPEG2TuneRequestFactory
        <PreserveSig()> _
        Function CreateTuneRequest(<[In]()> ByVal TuningSpace As ITuningSpace, <Out()> ByRef TuneRequest As IMPEG2TuneRequest) As Integer
    End Interface

    <ComImport(), System.Security.SuppressUnmanagedCodeSecurity(), Guid("FCD01846-0E19-11d3-9D8E-00C04F72D980"), InterfaceType(ComInterfaceType.InterfaceIsDual)> _
    Public Interface IComponents
        <PreserveSig()> _
        Function get_Count(<Out()> ByRef Count As Integer) As Integer

        <PreserveSig()> _
        Function get__NewEnum(<Out()> ByRef ppNewEnum As IEnumVARIANT) As Integer

        <PreserveSig()> _
        Function EnumComponents(<Out()> ByRef ppNewEnum As IEnumComponents) As Integer

        <PreserveSig()> _
        Function get_Item(<[In]()> ByVal varIndex As Object, <Out()> ByRef TuningSpace As IComponent) As Integer

        <PreserveSig()> _
        Function Add(<[In]()> ByVal Component As IComponent, <Out()> ByRef NewIndex As Object) As Integer

        <PreserveSig()> _
        Function Remove(<[In]()> ByVal Index As Object) As Integer

        <PreserveSig()> _
        Function Clone(<Out()> ByRef NewList As IComponents) As Integer
    End Interface

    <ComImport(), System.Security.SuppressUnmanagedCodeSecurity(), Guid("2A6E293C-2595-11d3-B64C-00C04F79498E"), InterfaceType(ComInterfaceType.InterfaceIsDual)> _
    Public Interface IAnalogTVTuningSpace
        Inherits ITuningSpace

        <PreserveSig()> _
        Shadows Function get_UniqueName(<Out(), MarshalAs(UnmanagedType.BStr)> ByRef Name As String) As Integer

        <PreserveSig()> _
        Shadows Function put_UniqueName(<[In](), MarshalAs(UnmanagedType.BStr)> ByVal Name As String) As Integer

        <PreserveSig()> _
        Shadows Function get_FriendlyName(<Out(), MarshalAs(UnmanagedType.BStr)> ByRef Name As String) As Integer

        <PreserveSig()> _
        Shadows Function put_FriendlyName(<[In](), MarshalAs(UnmanagedType.BStr)> ByVal Name As String) As Integer

        <PreserveSig()> _
        Shadows Function get_CLSID(<Out(), MarshalAs(UnmanagedType.BStr)> ByRef SpaceCLSID As String) As Integer

        <PreserveSig()> _
        Shadows Function get_NetworkType(<Out(), MarshalAs(UnmanagedType.BStr)> ByRef NetworkTypeGuid As String) As Integer

        <PreserveSig()> _
        Shadows Function put_NetworkType(<[In](), MarshalAs(UnmanagedType.BStr)> ByVal NetworkTypeGuid As String) As Integer

        <PreserveSig()> _
        Shadows Function get__NetworkType(<Out()> ByRef NetworkTypeGuid As Guid) As Integer

        <PreserveSig()> _
        Shadows Function put__NetworkType(<[In](), MarshalAs(UnmanagedType.LPStruct)> ByVal NetworkTypeGuid As Guid) As Integer

        <PreserveSig()> _
        Shadows Function CreateTuneRequest(<Out()> ByRef TuneRequest As ITuneRequest) As Integer

        ' IEnumGUID**
        <PreserveSig()> _
        Shadows Function EnumCategoryGUIDs(<Out(), MarshalAs(UnmanagedType.IUnknown)> ByRef ppEnum As Object) As Integer

        <PreserveSig()> _
        Shadows Function EnumDeviceMonikers(<Out()> ByRef ppEnum As IEnumMoniker) As Integer

        <PreserveSig()> _
        Shadows Function get_DefaultPreferredComponentTypes(<Out()> ByRef ComponentTypes As IComponentTypes) As Integer

        <PreserveSig()> _
        Shadows Function put_DefaultPreferredComponentTypes(<[In]()> ByVal NewComponentTypes As IComponentTypes) As Integer

        <PreserveSig()> _
        Shadows Function get_FrequencyMapping(<Out(), MarshalAs(UnmanagedType.BStr)> ByRef pMapping As String) As Integer

        <PreserveSig()> _
        Shadows Function put_FrequencyMapping(<[In](), MarshalAs(UnmanagedType.BStr)> ByVal Mapping As String) As Integer

        <PreserveSig()> _
        Shadows Function get_DefaultLocator(<Out()> ByRef LocatorVal As ILocator) As Integer

        <PreserveSig()> _
        Shadows Function put_DefaultLocator(<[In]()> ByVal LocatorVal As ILocator) As Integer

        <PreserveSig()> _
        Shadows Function Clone(<Out()> ByRef NewTS As ITuningSpace) As Integer

        <PreserveSig()> _
        Function get_MinChannel(<Out()> ByRef MinChannelVal As Integer) As Integer

        <PreserveSig()> _
        Function put_MinChannel(<[In]()> ByVal NewMinChannelVal As Integer) As Integer

        <PreserveSig()> _
        Function get_MaxChannel(<Out()> ByRef MaxChannelVal As Integer) As Integer

        <PreserveSig()> _
        Function put_MaxChannel(<[In]()> ByVal NewMaxChannelVal As Integer) As Integer

        <PreserveSig()> _
        Function get_InputType(<Out()> ByRef InputTypeVal As TunerInputType) As Integer

        <PreserveSig()> _
        Function put_InputType(<[In]()> ByVal NewInputTypeVal As TunerInputType) As Integer

        <PreserveSig()> _
        Function get_CountryCode(<Out()> ByRef CountryCodeVal As Integer) As Integer

        <PreserveSig()> _
        Function put_CountryCode(<[In]()> ByVal NewCountryCodeVal As Integer) As Integer
    End Interface

    <ComImport(), System.Security.SuppressUnmanagedCodeSecurity(), Guid("0369B4E1-45B6-11d3-B650-00C04F79498E"), InterfaceType(ComInterfaceType.InterfaceIsDual)> _
    Public Interface IATSCChannelTuneRequest
        Inherits IChannelTuneRequest

        <PreserveSig()> _
        Shadows Function get_TuningSpace(<Out()> ByRef TuningSpace As ITuningSpace) As Integer

        <PreserveSig()> _
        Shadows Function get_Components(<Out()> ByRef Components As IComponents) As Integer

        <PreserveSig()> _
        Shadows Function Clone(<Out()> ByRef NewTuneRequest As ITuneRequest) As Integer

        <PreserveSig()> _
        Shadows Function get_Locator(<Out()> ByRef Locator As ILocator) As Integer

        <PreserveSig()> _
        Shadows Function put_Locator(<[In]()> ByVal Locator As ILocator) As Integer

        <PreserveSig()> _
        Shadows Function get_Channel(<Out()> ByRef Channel As Integer) As Integer

        <PreserveSig()> _
        Shadows Function put_Channel(<[In]()> ByVal Channel As Integer) As Integer

        <PreserveSig()> _
        Function get_MinorChannel(<Out()> ByRef MinorChannel As Integer) As Integer

        <PreserveSig()> _
        Function put_MinorChannel(<[In]()> ByVal MinorChannel As Integer) As Integer
    End Interface

    <ComImport(), System.Security.SuppressUnmanagedCodeSecurity(), Guid("FC189E4D-7BD4-4125-B3B3-3A76A332CC96"), InterfaceType(ComInterfaceType.InterfaceIsDual)> _
    Public Interface IATSCComponentType
        Inherits IMPEG2ComponentType

        <PreserveSig()> _
        Shadows Function get_Category(<Out()> ByRef Category As ComponentCategory) As Integer

        <PreserveSig()> _
        Shadows Function put_Category(<[In]()> ByVal Category As ComponentCategory) As Integer

        <PreserveSig()> _
        Shadows Function get_MediaMajorType(<Out(), MarshalAs(UnmanagedType.BStr)> ByRef MediaMajorType As String) As Integer

        <PreserveSig()> _
        Shadows Function put_MediaMajorType(<[In](), MarshalAs(UnmanagedType.BStr)> ByVal MediaMajorType As String) As Integer

        <PreserveSig()> _
        Shadows Function get__MediaMajorType(<Out()> ByRef MediaMajorType As Guid) As Integer

        <PreserveSig()> _
        Shadows Function put__MediaMajorType(<[In](), MarshalAs(UnmanagedType.LPStruct)> ByVal MediaMajorType As Guid) As Integer

        <PreserveSig()> _
        Shadows Function get_MediaSubType(<Out(), MarshalAs(UnmanagedType.BStr)> ByRef MediaSubType As String) As Integer

        <PreserveSig()> _
        Shadows Function put_MediaSubType(<[In](), MarshalAs(UnmanagedType.BStr)> ByVal MediaSubType As String) As Integer

        <PreserveSig()> _
        Shadows Function get__MediaSubType(<Out()> ByRef MediaSubType As Guid) As Integer

        <PreserveSig()> _
        Shadows Function put__MediaSubType(<[In](), MarshalAs(UnmanagedType.LPStruct)> ByVal MediaSubType As Guid) As Integer

        <PreserveSig()> _
        Shadows Function get_MediaFormatType(<Out(), MarshalAs(UnmanagedType.BStr)> ByRef MediaFormatType As String) As Integer

        <PreserveSig()> _
        Shadows Function put_MediaFormatType(<[In](), MarshalAs(UnmanagedType.BStr)> ByVal MediaFormatType As String) As Integer

        <PreserveSig()> _
        Shadows Function get__MediaFormatType(<Out()> ByRef MediaFormatType As Guid) As Integer

        <PreserveSig()> _
        Shadows Function put__MediaFormatType(<[In](), MarshalAs(UnmanagedType.LPStruct)> ByVal MediaFormatType As Guid) As Integer

        <PreserveSig()> _
        Shadows Function get_MediaType(<Out()> ByVal MediaType As AMMediaType) As Integer

        <PreserveSig()> _
        Shadows Function put_MediaType(<[In]()> ByVal MediaType As AMMediaType) As Integer

        <PreserveSig()> _
        Shadows Function Clone(<Out()> ByRef NewCT As IComponentType) As Integer

        <PreserveSig()> _
        Shadows Function get_LangID(<Out()> ByRef LangID As Integer) As Integer

        <PreserveSig()> _
        Shadows Function put_LangID(<[In]()> ByVal LangID As Integer) As Integer

        <PreserveSig()> _
        Shadows Function get_StreamType(<Out()> ByRef MP2StreamType As MPEG2StreamType) As Integer

        <PreserveSig()> _
        Shadows Function put_StreamType(<[In]()> ByVal MP2StreamType As MPEG2StreamType) As Integer

        <PreserveSig()> _
        Function get_Flags(<Out()> ByRef Flags As ATSCComponentTypeFlags) As Integer

        <PreserveSig()> _
        Function put_Flags(<[In]()> ByVal Flags As ATSCComponentTypeFlags) As Integer
    End Interface

    <ComImport(), System.Security.SuppressUnmanagedCodeSecurity(), Guid("BF8D986F-8C2B-4131-94D7-4D3D9FCC21EF"), InterfaceType(ComInterfaceType.InterfaceIsDual)> _
    Public Interface IATSCLocator
        Inherits IDigitalLocator

        <PreserveSig()> _
        Shadows Function get_CarrierFrequency(<Out()> ByRef Frequency As Integer) As Integer

        <PreserveSig()> _
        Shadows Function put_CarrierFrequency(<[In]()> ByVal Frequency As Integer) As Integer

        <PreserveSig()> _
        Shadows Function get_InnerFEC(<Out()> ByRef FEC As FECMethod) As Integer

        <PreserveSig()> _
        Shadows Function put_InnerFEC(<[In]()> ByVal FEC As FECMethod) As Integer

        <PreserveSig()> _
        Shadows Function get_InnerFECRate(<Out()> ByRef FEC As BinaryConvolutionCodeRate) As Integer

        <PreserveSig()> _
        Shadows Function put_InnerFECRate(<[In]()> ByVal FEC As BinaryConvolutionCodeRate) As Integer

        <PreserveSig()> _
        Shadows Function get_OuterFEC(<Out()> ByRef FEC As FECMethod) As Integer

        <PreserveSig()> _
        Shadows Function put_OuterFEC(<[In]()> ByVal FEC As FECMethod) As Integer

        <PreserveSig()> _
        Shadows Function get_OuterFECRate(<Out()> ByRef FEC As BinaryConvolutionCodeRate) As Integer

        <PreserveSig()> _
        Shadows Function put_OuterFECRate(<[In]()> ByVal FEC As BinaryConvolutionCodeRate) As Integer

        <PreserveSig()> _
        Shadows Function get_Modulation(<Out()> ByRef Modulation As ModulationType) As Integer

        <PreserveSig()> _
        Shadows Function put_Modulation(<[In]()> ByVal Modulation As ModulationType) As Integer

        <PreserveSig()> _
        Shadows Function get_SymbolRate(<Out()> ByRef Rate As Integer) As Integer

        <PreserveSig()> _
        Shadows Function put_SymbolRate(<[In]()> ByVal Rate As Integer) As Integer

        <PreserveSig()> _
        Shadows Function Clone(<Out()> ByRef NewLocator As ILocator) As Integer

        <PreserveSig()> _
        Function get_PhysicalChannel(<Out()> ByRef PhysicalChannel As Integer) As Integer

        <PreserveSig()> _
        Function put_PhysicalChannel(<[In]()> ByVal PhysicalChannel As Integer) As Integer

        <PreserveSig()> _
        Function get_TSID(<Out()> ByRef TSID As Integer) As Integer

        <PreserveSig()> _
        Function put_TSID(<[In]()> ByVal TSID As Integer) As Integer
    End Interface

    <ComImport(), System.Security.SuppressUnmanagedCodeSecurity(), Guid("0369B4E2-45B6-11d3-B650-00C04F79498E"), InterfaceType(ComInterfaceType.InterfaceIsDual)> _
    Public Interface IATSCTuningSpace
        Inherits IAnalogTVTuningSpace

        <PreserveSig()> _
        Shadows Function get_UniqueName(<Out(), MarshalAs(UnmanagedType.BStr)> ByRef Name As String) As Integer

        <PreserveSig()> _
        Shadows Function put_UniqueName(<[In](), MarshalAs(UnmanagedType.BStr)> ByVal Name As String) As Integer

        <PreserveSig()> _
        Shadows Function get_FriendlyName(<Out(), MarshalAs(UnmanagedType.BStr)> ByRef Name As String) As Integer

        <PreserveSig()> _
        Shadows Function put_FriendlyName(<[In](), MarshalAs(UnmanagedType.BStr)> ByVal Name As String) As Integer

        <PreserveSig()> _
        Shadows Function get_CLSID(<Out(), MarshalAs(UnmanagedType.BStr)> ByRef SpaceCLSID As String) As Integer

        <PreserveSig()> _
        Shadows Function get_NetworkType(<Out(), MarshalAs(UnmanagedType.BStr)> ByRef NetworkTypeGuid As String) As Integer

        <PreserveSig()> _
        Shadows Function put_NetworkType(<[In](), MarshalAs(UnmanagedType.BStr)> ByVal NetworkTypeGuid As String) As Integer

        <PreserveSig()> _
        Shadows Function get__NetworkType(<Out()> ByRef NetworkTypeGuid As Guid) As Integer

        <PreserveSig()> _
        Shadows Function put__NetworkType(<[In](), MarshalAs(UnmanagedType.LPStruct)> ByVal NetworkTypeGuid As Guid) As Integer

        <PreserveSig()> _
        Shadows Function CreateTuneRequest(<Out()> ByRef TuneRequest As ITuneRequest) As Integer

        ' IEnumGUID**
        <PreserveSig()> _
        Shadows Function EnumCategoryGUIDs(<Out(), MarshalAs(UnmanagedType.IUnknown)> ByRef ppEnum As Object) As Integer

        <PreserveSig()> _
        Shadows Function EnumDeviceMonikers(<Out()> ByRef ppEnum As IEnumMoniker) As Integer

        <PreserveSig()> _
        Shadows Function get_DefaultPreferredComponentTypes(<Out()> ByRef ComponentTypes As IComponentTypes) As Integer

        <PreserveSig()> _
        Shadows Function put_DefaultPreferredComponentTypes(<[In]()> ByVal NewComponentTypes As IComponentTypes) As Integer

        <PreserveSig()> _
        Shadows Function get_FrequencyMapping(<Out(), MarshalAs(UnmanagedType.BStr)> ByRef pMapping As String) As Integer

        <PreserveSig()> _
        Shadows Function put_FrequencyMapping(<[In](), MarshalAs(UnmanagedType.BStr)> ByVal Mapping As String) As Integer

        <PreserveSig()> _
        Shadows Function get_DefaultLocator(<Out()> ByRef LocatorVal As ILocator) As Integer

        <PreserveSig()> _
        Shadows Function put_DefaultLocator(<[In]()> ByVal LocatorVal As ILocator) As Integer

        <PreserveSig()> _
        Shadows Function Clone(<Out()> ByRef NewTS As ITuningSpace) As Integer

        <PreserveSig()> _
        Shadows Function get_MinChannel(<Out()> ByRef MinChannelVal As Integer) As Integer

        <PreserveSig()> _
        Shadows Function put_MinChannel(<[In]()> ByVal NewMinChannelVal As Integer) As Integer

        <PreserveSig()> _
        Shadows Function get_MaxChannel(<Out()> ByRef MaxChannelVal As Integer) As Integer

        <PreserveSig()> _
        Shadows Function put_MaxChannel(<[In]()> ByVal NewMaxChannelVal As Integer) As Integer

        <PreserveSig()> _
        Shadows Function get_InputType(<Out()> ByRef InputTypeVal As TunerInputType) As Integer

        <PreserveSig()> _
        Shadows Function put_InputType(<[In]()> ByVal NewInputTypeVal As TunerInputType) As Integer

        <PreserveSig()> _
        Shadows Function get_CountryCode(<Out()> ByRef CountryCodeVal As Integer) As Integer

        <PreserveSig()> _
        Shadows Function put_CountryCode(<[In]()> ByVal NewCountryCodeVal As Integer) As Integer

        <PreserveSig()> _
        Function get_MinMinorChannel(<Out()> ByRef MinMinorChannelVal As Integer) As Integer

        <PreserveSig()> _
        Function put_MinMinorChannel(<[In]()> ByVal NewMinMinorChannelVal As Integer) As Integer

        <PreserveSig()> _
        Function get_MaxMinorChannel(<Out()> ByRef MaxMinorChannelVal As Integer) As Integer

        <PreserveSig()> _
        Function put_MaxMinorChannel(<[In]()> ByVal NewMaxMinorChannelVal As Integer) As Integer

        <PreserveSig()> _
        Function get_MinPhysicalChannel(<Out()> ByRef MinPhysicalChannelVal As Integer) As Integer

        <PreserveSig()> _
        Function put_MinPhysicalChannel(<[In]()> ByVal NewMinPhysicalChannelVal As Integer) As Integer

        <PreserveSig()> _
        Function get_MaxPhysicalChannel(<Out()> ByRef MaxPhysicalChannelVal As Integer) As Integer

        <PreserveSig()> _
        Function put_MaxPhysicalChannel(<[In]()> ByVal NewMaxPhysicalChannelVal As Integer) As Integer
    End Interface

    <ComImport(), System.Security.SuppressUnmanagedCodeSecurity(), Guid("0369B4E0-45B6-11d3-B650-00C04F79498E"), InterfaceType(ComInterfaceType.InterfaceIsDual)> _
    Public Interface IChannelTuneRequest
        Inherits ITuneRequest

        <PreserveSig()> _
        Shadows Function get_TuningSpace(<Out()> ByRef TuningSpace As ITuningSpace) As Integer

        <PreserveSig()> _
        Shadows Function get_Components(<Out()> ByRef Components As IComponents) As Integer

        <PreserveSig()> _
        Shadows Function Clone(<Out()> ByRef NewTuneRequest As ITuneRequest) As Integer

        <PreserveSig()> _
        Shadows Function get_Locator(<Out()> ByRef Locator As ILocator) As Integer

        <PreserveSig()> _
        Shadows Function put_Locator(<[In]()> ByVal Locator As ILocator) As Integer

        <PreserveSig()> _
        Function get_Channel(<Out()> ByRef Channel As Integer) As Integer

        <PreserveSig()> _
        Function put_Channel(<[In]()> ByVal Channel As Integer) As Integer
    End Interface

    <ComImport(), System.Security.SuppressUnmanagedCodeSecurity(), Guid("B874C8BA-0FA2-11d3-9D8E-00C04F72D980"), InterfaceType(ComInterfaceType.InterfaceIsDual)> _
    Public Interface ILanguageComponentType
        Inherits IComponentType

        <PreserveSig()> _
        Shadows Function get_Category(<Out()> ByRef Category As ComponentCategory) As Integer

        <PreserveSig()> _
        Shadows Function put_Category(<[In]()> ByVal Category As ComponentCategory) As Integer

        <PreserveSig()> _
        Shadows Function get_MediaMajorType(<Out(), MarshalAs(UnmanagedType.BStr)> ByRef MediaMajorType As String) As Integer

        <PreserveSig()> _
        Shadows Function put_MediaMajorType(<[In](), MarshalAs(UnmanagedType.BStr)> ByVal MediaMajorType As String) As Integer

        <PreserveSig()> _
        Shadows Function get__MediaMajorType(<Out()> ByRef MediaMajorType As Guid) As Integer

        <PreserveSig()> _
        Shadows Function put__MediaMajorType(<[In](), MarshalAs(UnmanagedType.LPStruct)> ByVal MediaMajorType As Guid) As Integer

        <PreserveSig()> _
        Shadows Function get_MediaSubType(<Out(), MarshalAs(UnmanagedType.BStr)> ByRef MediaSubType As String) As Integer

        <PreserveSig()> _
        Shadows Function put_MediaSubType(<[In](), MarshalAs(UnmanagedType.BStr)> ByVal MediaSubType As String) As Integer

        <PreserveSig()> _
        Shadows Function get__MediaSubType(<Out()> ByRef MediaSubType As Guid) As Integer

        <PreserveSig()> _
        Shadows Function put__MediaSubType(<[In](), MarshalAs(UnmanagedType.LPStruct)> ByVal MediaSubType As Guid) As Integer

        <PreserveSig()> _
        Shadows Function get_MediaFormatType(<Out(), MarshalAs(UnmanagedType.BStr)> ByRef MediaFormatType As String) As Integer

        <PreserveSig()> _
        Shadows Function put_MediaFormatType(<[In](), MarshalAs(UnmanagedType.BStr)> ByVal MediaFormatType As String) As Integer

        <PreserveSig()> _
        Shadows Function get__MediaFormatType(<Out()> ByRef MediaFormatType As Guid) As Integer

        <PreserveSig()> _
        Shadows Function put__MediaFormatType(<[In](), MarshalAs(UnmanagedType.LPStruct)> ByVal MediaFormatType As Guid) As Integer

        <PreserveSig()> _
        Shadows Function get_MediaType(<Out()> ByVal MediaType As AMMediaType) As Integer

        <PreserveSig()> _
        Shadows Function put_MediaType(<[In]()> ByVal MediaType As AMMediaType) As Integer

        <PreserveSig()> _
        Shadows Function Clone(<Out()> ByRef NewCT As IComponentType) As Integer

        <PreserveSig()> _
        Function get_LangID(<Out()> ByRef LangID As Integer) As Integer

        <PreserveSig()> _
        Function put_LangID(<[In]()> ByVal LangID As Integer) As Integer
    End Interface

    <ComImport(), System.Security.SuppressUnmanagedCodeSecurity(), Guid("1493E353-1EB6-473c-802D-8E6B8EC9D2A9"), InterfaceType(ComInterfaceType.InterfaceIsDual)> _
    Public Interface IMPEG2Component
        Inherits IComponent

        <PreserveSig()> _
        Shadows Function get_Type(<Out()> ByRef CT As IComponentType) As Integer

        <PreserveSig()> _
        Shadows Function put_Type(<[In]()> ByVal CT As IComponentType) As Integer

        <PreserveSig()> _
        Shadows Function get_DescLangID(<Out()> ByRef LangID As Integer) As Integer

        <PreserveSig()> _
        Shadows Function put_DescLangID(<[In]()> ByVal LangID As Integer) As Integer

        <PreserveSig()> _
        Shadows Function get_Status(<Out()> ByRef Status As ComponentStatus) As Integer

        <PreserveSig()> _
        Shadows Function put_Status(<[In]()> ByVal Status As ComponentStatus) As Integer

        <PreserveSig()> _
        Shadows Function get_Description(<Out(), MarshalAs(UnmanagedType.BStr)> ByRef Description As String) As Integer

        <PreserveSig()> _
        Shadows Function put_Description(<[In](), MarshalAs(UnmanagedType.BStr)> ByVal Description As String) As Integer

        <PreserveSig()> _
        Shadows Function Clone(<Out()> ByRef NewComponent As IComponent) As Integer

        <PreserveSig()> _
        Function get_PID(<Out()> ByRef PID As Integer) As Integer

        <PreserveSig()> _
        Function put_PID(<[In]()> ByVal PID As Integer) As Integer

        <PreserveSig()> _
        Function get_PCRPID(<Out()> ByRef PCRPID As Integer) As Integer

        <PreserveSig()> _
        Function put_PCRPID(<[In]()> ByVal PCRPID As Integer) As Integer

        <PreserveSig()> _
        Function get_ProgramNumber(<Out()> ByRef ProgramNumber As Integer) As Integer

        <PreserveSig()> _
        Function put_ProgramNumber(<[In]()> ByVal ProgramNumber As Integer) As Integer
    End Interface

    <ComImport(), System.Security.SuppressUnmanagedCodeSecurity(), Guid("2C073D84-B51C-48c9-AA9F-68971E1F6E38"), InterfaceType(ComInterfaceType.InterfaceIsDual)> _
    Public Interface IMPEG2ComponentType
        Inherits ILanguageComponentType

        <PreserveSig()> _
        Shadows Function get_Category(<Out()> ByRef Category As ComponentCategory) As Integer

        <PreserveSig()> _
        Shadows Function put_Category(<[In]()> ByVal Category As ComponentCategory) As Integer

        <PreserveSig()> _
        Shadows Function get_MediaMajorType(<Out(), MarshalAs(UnmanagedType.BStr)> ByRef MediaMajorType As String) As Integer

        <PreserveSig()> _
        Shadows Function put_MediaMajorType(<[In](), MarshalAs(UnmanagedType.BStr)> ByVal MediaMajorType As String) As Integer

        <PreserveSig()> _
        Shadows Function get__MediaMajorType(<Out()> ByRef MediaMajorType As Guid) As Integer

        <PreserveSig()> _
        Shadows Function put__MediaMajorType(<[In](), MarshalAs(UnmanagedType.LPStruct)> ByVal MediaMajorType As Guid) As Integer

        <PreserveSig()> _
        Shadows Function get_MediaSubType(<Out(), MarshalAs(UnmanagedType.BStr)> ByRef MediaSubType As String) As Integer

        <PreserveSig()> _
        Shadows Function put_MediaSubType(<[In](), MarshalAs(UnmanagedType.BStr)> ByVal MediaSubType As String) As Integer

        <PreserveSig()> _
        Shadows Function get__MediaSubType(<Out()> ByRef MediaSubType As Guid) As Integer

        <PreserveSig()> _
        Shadows Function put__MediaSubType(<[In](), MarshalAs(UnmanagedType.LPStruct)> ByVal MediaSubType As Guid) As Integer

        <PreserveSig()> _
        Shadows Function get_MediaFormatType(<Out(), MarshalAs(UnmanagedType.BStr)> ByRef MediaFormatType As String) As Integer

        <PreserveSig()> _
        Shadows Function put_MediaFormatType(<[In](), MarshalAs(UnmanagedType.BStr)> ByVal MediaFormatType As String) As Integer

        <PreserveSig()> _
        Shadows Function get__MediaFormatType(<Out()> ByRef MediaFormatType As Guid) As Integer

        <PreserveSig()> _
        Shadows Function put__MediaFormatType(<[In](), MarshalAs(UnmanagedType.LPStruct)> ByVal MediaFormatType As Guid) As Integer

        <PreserveSig()> _
        Shadows Function get_MediaType(<Out()> ByVal MediaType As AMMediaType) As Integer

        <PreserveSig()> _
        Shadows Function put_MediaType(<[In]()> ByVal MediaType As AMMediaType) As Integer

        <PreserveSig()> _
        Shadows Function Clone(<Out()> ByRef NewCT As IComponentType) As Integer

        <PreserveSig()> _
        Shadows Function get_LangID(<Out()> ByRef LangID As Integer) As Integer

        <PreserveSig()> _
        Shadows Function put_LangID(<[In]()> ByVal LangID As Integer) As Integer

        <PreserveSig()> _
        Function get_StreamType(<Out()> ByRef MP2StreamType As MPEG2StreamType) As Integer

        <PreserveSig()> _
        Function put_StreamType(<[In]()> ByVal MP2StreamType As MPEG2StreamType) As Integer
    End Interface

    <ComImport(), System.Security.SuppressUnmanagedCodeSecurity(), Guid("E48244B8-7E17-4f76-A763-5090FF1E2F30"), InterfaceType(ComInterfaceType.InterfaceIsDual)> _
    Public Interface IAuxInTuningSpace
        Inherits ITuningSpace

        <PreserveSig()> _
        Shadows Function get_UniqueName(<Out(), MarshalAs(UnmanagedType.BStr)> ByRef Name As String) As Integer

        <PreserveSig()> _
        Shadows Function put_UniqueName(<[In](), MarshalAs(UnmanagedType.BStr)> ByVal Name As String) As Integer

        <PreserveSig()> _
        Shadows Function get_FriendlyName(<Out(), MarshalAs(UnmanagedType.BStr)> ByRef Name As String) As Integer

        <PreserveSig()> _
        Shadows Function put_FriendlyName(<[In](), MarshalAs(UnmanagedType.BStr)> ByVal Name As String) As Integer

        <PreserveSig()> _
        Shadows Function get_CLSID(<Out(), MarshalAs(UnmanagedType.BStr)> ByRef SpaceCLSID As String) As Integer

        <PreserveSig()> _
        Shadows Function get_NetworkType(<Out(), MarshalAs(UnmanagedType.BStr)> ByRef NetworkTypeGuid As String) As Integer

        <PreserveSig()> _
        Shadows Function put_NetworkType(<[In](), MarshalAs(UnmanagedType.BStr)> ByVal NetworkTypeGuid As String) As Integer

        <PreserveSig()> _
        Shadows Function get__NetworkType(<Out()> ByRef NetworkTypeGuid As Guid) As Integer

        <PreserveSig()> _
        Shadows Function put__NetworkType(<[In](), MarshalAs(UnmanagedType.LPStruct)> ByVal NetworkTypeGuid As Guid) As Integer

        <PreserveSig()> _
        Shadows Function CreateTuneRequest(<Out()> ByRef TuneRequest As ITuneRequest) As Integer

        ' IEnumGUID**
        <PreserveSig()> _
        Shadows Function EnumCategoryGUIDs(<Out(), MarshalAs(UnmanagedType.IUnknown)> ByRef ppEnum As Object) As Integer

        <PreserveSig()> _
        Shadows Function EnumDeviceMonikers(<Out()> ByRef ppEnum As IEnumMoniker) As Integer

        <PreserveSig()> _
        Shadows Function get_DefaultPreferredComponentTypes(<Out()> ByRef ComponentTypes As IComponentTypes) As Integer

        <PreserveSig()> _
        Shadows Function put_DefaultPreferredComponentTypes(<[In]()> ByVal NewComponentTypes As IComponentTypes) As Integer

        <PreserveSig()> _
        Shadows Function get_FrequencyMapping(<Out(), MarshalAs(UnmanagedType.BStr)> ByRef pMapping As String) As Integer

        <PreserveSig()> _
        Shadows Function put_FrequencyMapping(<[In](), MarshalAs(UnmanagedType.BStr)> ByVal Mapping As String) As Integer

        <PreserveSig()> _
        Shadows Function get_DefaultLocator(<Out()> ByRef LocatorVal As ILocator) As Integer

        <PreserveSig()> _
        Shadows Function put_DefaultLocator(<[In]()> ByVal LocatorVal As ILocator) As Integer

        <PreserveSig()> _
        Shadows Function Clone(<Out()> ByRef NewTS As ITuningSpace) As Integer

    End Interface

    <ComImport(), System.Security.SuppressUnmanagedCodeSecurity(), Guid("6E42F36E-1DD2-43c4-9F78-69D25AE39034"), InterfaceType(ComInterfaceType.InterfaceIsDual)> _
    Public Interface IDVBCLocator
        Inherits IDigitalLocator

        <PreserveSig()> _
        Shadows Function get_CarrierFrequency(<Out()> ByRef Frequency As Integer) As Integer

        <PreserveSig()> _
        Shadows Function put_CarrierFrequency(<[In]()> ByVal Frequency As Integer) As Integer

        <PreserveSig()> _
        Shadows Function get_InnerFEC(<Out()> ByRef FEC As FECMethod) As Integer

        <PreserveSig()> _
        Shadows Function put_InnerFEC(<[In]()> ByVal FEC As FECMethod) As Integer

        <PreserveSig()> _
        Shadows Function get_InnerFECRate(<Out()> ByRef FEC As BinaryConvolutionCodeRate) As Integer

        <PreserveSig()> _
        Shadows Function put_InnerFECRate(<[In]()> ByVal FEC As BinaryConvolutionCodeRate) As Integer

        <PreserveSig()> _
        Shadows Function get_OuterFEC(<Out()> ByRef FEC As FECMethod) As Integer

        <PreserveSig()> _
        Shadows Function put_OuterFEC(<[In]()> ByVal FEC As FECMethod) As Integer

        <PreserveSig()> _
        Shadows Function get_OuterFECRate(<Out()> ByRef FEC As BinaryConvolutionCodeRate) As Integer

        <PreserveSig()> _
        Shadows Function put_OuterFECRate(<[In]()> ByVal FEC As BinaryConvolutionCodeRate) As Integer

        <PreserveSig()> _
        Shadows Function get_Modulation(<Out()> ByRef Modulation As ModulationType) As Integer

        <PreserveSig()> _
        Shadows Function put_Modulation(<[In]()> ByVal Modulation As ModulationType) As Integer

        <PreserveSig()> _
        Shadows Function get_SymbolRate(<Out()> ByRef Rate As Integer) As Integer

        <PreserveSig()> _
        Shadows Function put_SymbolRate(<[In]()> ByVal Rate As Integer) As Integer

        <PreserveSig()> _
        Shadows Function Clone(<Out()> ByRef NewLocator As ILocator) As Integer

    End Interface

    <ComImport(), System.Security.SuppressUnmanagedCodeSecurity(), Guid("1B9D5FC3-5BBC-4b6c-BB18-B9D10E3EEEBF"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)> _
    Public Interface IMPEG2TuneRequestSupport
    End Interface

    <ComImport(), System.Security.SuppressUnmanagedCodeSecurity(), Guid("3B21263F-26E8-489d-AAC4-924F7EFD9511"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)> _
    Public Interface IBroadcastEvent
        <PreserveSig()> _
        Function Fire(ByVal EventID As Guid) As Integer
    End Interface

    <ComImport(), System.Security.SuppressUnmanagedCodeSecurity(), Guid("B34505E0-2F0E-497b-80BC-D43F3B24ED7F"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)> _
    Public Interface IBDAComparable
        <PreserveSig()> _
        Function CompareExact(<[In](), MarshalAs(UnmanagedType.IDispatch)> ByVal CompareTo As Object, <Out()> ByRef Result As Integer) As Integer

        <PreserveSig()> _
        Function CompareEquivalent(<[In](), MarshalAs(UnmanagedType.IDispatch)> ByVal CompareTo As Object, <[In]()> ByVal dwFlags As BDACompFlags, <Out()> ByRef Result As Integer) As Integer

        <PreserveSig()> _
        Function HashExact(<Out()> ByRef Result As Long) As Integer

        <PreserveSig()> _
        Function HashExactIncremental(<[In]()> ByVal PartialResult As Long, <Out()> ByRef Result As Long) As Integer

        <PreserveSig()> _
        Function HashEquivalent(<[In]()> ByVal dwFlags As BDACompFlags, <Out()> ByRef Result As Long) As Integer

        <PreserveSig()> _
        Function HashEquivalentIncremental(<[In]()> ByVal PartialResult As Long, <[In]()> ByVal dwFlags As BDACompFlags, <Out()> ByRef Result As Long) As Integer
    End Interface

    <ComImport(), System.Security.SuppressUnmanagedCodeSecurity(), Guid("19B595D8-839A-47F0-96DF-4F194F3C768C"), InterfaceType(ComInterfaceType.InterfaceIsDual)> _
    Public Interface IDigitalLocator
        Inherits ILocator

        <PreserveSig()> _
        Shadows Function get_CarrierFrequency(<Out()> ByRef Frequency As Integer) As Integer

        <PreserveSig()> _
        Shadows Function put_CarrierFrequency(<[In]()> ByVal Frequency As Integer) As Integer

        <PreserveSig()> _
        Shadows Function get_InnerFEC(<Out()> ByRef FEC As FECMethod) As Integer

        <PreserveSig()> _
        Shadows Function put_InnerFEC(<[In]()> ByVal FEC As FECMethod) As Integer

        <PreserveSig()> _
        Shadows Function get_InnerFECRate(<Out()> ByRef FEC As BinaryConvolutionCodeRate) As Integer

        <PreserveSig()> _
        Shadows Function put_InnerFECRate(<[In]()> ByVal FEC As BinaryConvolutionCodeRate) As Integer

        <PreserveSig()> _
        Shadows Function get_OuterFEC(<Out()> ByRef FEC As FECMethod) As Integer

        <PreserveSig()> _
        Shadows Function put_OuterFEC(<[In]()> ByVal FEC As FECMethod) As Integer

        <PreserveSig()> _
        Shadows Function get_OuterFECRate(<Out()> ByRef FEC As BinaryConvolutionCodeRate) As Integer

        <PreserveSig()> _
        Shadows Function put_OuterFECRate(<[In]()> ByVal FEC As BinaryConvolutionCodeRate) As Integer

        <PreserveSig()> _
        Shadows Function get_Modulation(<Out()> ByRef Modulation As ModulationType) As Integer

        <PreserveSig()> _
        Shadows Function put_Modulation(<[In]()> ByVal Modulation As ModulationType) As Integer

        <PreserveSig()> _
        Shadows Function get_SymbolRate(<Out()> ByRef Rate As Integer) As Integer

        <PreserveSig()> _
        Shadows Function put_SymbolRate(<[In]()> ByVal Rate As Integer) As Integer

        <PreserveSig()> _
        Shadows Function Clone(<Out()> ByRef NewLocator As ILocator) As Integer

    End Interface

    <ComImport(), System.Security.SuppressUnmanagedCodeSecurity(), Guid("3d9e3887-1929-423f-8021-43682de95448"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)> _
    Public Interface IBroadcastEventEx
        Inherits IBroadcastEvent

        <PreserveSig()> _
        Shadows Function Fire(ByVal EventID As Guid) As Integer

        <PreserveSig()> _
        Function FireEx(<[In]()> ByVal EventID As Guid, <[In]()> ByVal Param1 As Integer, <[In]()> ByVal Param2 As Integer, <[In]()> ByVal Param3 As Integer, <[In]()> ByVal Param4 As Integer) As Integer
    End Interface

    <ComImport(), System.Security.SuppressUnmanagedCodeSecurity(), Guid("E60DFA45-8D56-4e65-A8AB-D6BE9412C249"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)> _
    Public Interface ITunerCap
        <PreserveSig()> _
        Function get_SupportedNetworkTypes(<[In]()> ByVal ulcNetworkTypesMax As Integer, <Out()> ByRef pulcNetworkTypes As Integer, <[In](), Out(), MarshalAs(UnmanagedType.LPArray, ArraySubType:=UnmanagedType.Struct)> ByVal pguidNetworkTypes As Guid()) As Integer

        <PreserveSig()> _
        Function get_SupportedVideoFormats(<Out()> ByRef pulAMTunerModeType As AMTunerModeType, <Out()> ByRef pulAnalogVideoStandard As AnalogVideoStandard) As Integer

        <PreserveSig()> _
        Function get_AuxInputCount(<Out()> ByRef pulCompositeCount As Integer, <Out()> ByRef pulSvideoCount As Integer) As Integer
    End Interface

    <ComImport(), System.Security.SuppressUnmanagedCodeSecurity(), Guid("2A6E293B-2595-11d3-B64C-00C04F79498E"), InterfaceType(ComInterfaceType.InterfaceIsDual)> _
    Public Interface IAnalogRadioTuningSpace
        Inherits ITuningSpace

        <PreserveSig()> _
        Shadows Function get_UniqueName(<Out(), MarshalAs(UnmanagedType.BStr)> ByRef Name As String) As Integer

        <PreserveSig()> _
        Shadows Function put_UniqueName(<[In](), MarshalAs(UnmanagedType.BStr)> ByVal Name As String) As Integer

        <PreserveSig()> _
        Shadows Function get_FriendlyName(<Out(), MarshalAs(UnmanagedType.BStr)> ByRef Name As String) As Integer

        <PreserveSig()> _
        Shadows Function put_FriendlyName(<[In](), MarshalAs(UnmanagedType.BStr)> ByVal Name As String) As Integer

        <PreserveSig()> _
        Shadows Function get_CLSID(<Out(), MarshalAs(UnmanagedType.BStr)> ByRef SpaceCLSID As String) As Integer

        <PreserveSig()> _
        Shadows Function get_NetworkType(<Out(), MarshalAs(UnmanagedType.BStr)> ByRef NetworkTypeGuid As String) As Integer

        <PreserveSig()> _
        Shadows Function put_NetworkType(<[In](), MarshalAs(UnmanagedType.BStr)> ByVal NetworkTypeGuid As String) As Integer

        <PreserveSig()> _
        Shadows Function get__NetworkType(<Out()> ByRef NetworkTypeGuid As Guid) As Integer

        <PreserveSig()> _
        Shadows Function put__NetworkType(<[In](), MarshalAs(UnmanagedType.LPStruct)> ByVal NetworkTypeGuid As Guid) As Integer

        <PreserveSig()> _
        Shadows Function CreateTuneRequest(<Out()> ByRef TuneRequest As ITuneRequest) As Integer

        ' IEnumGUID**
        <PreserveSig()> _
        Shadows Function EnumCategoryGUIDs(<Out(), MarshalAs(UnmanagedType.IUnknown)> ByRef ppEnum As Object) As Integer

        <PreserveSig()> _
        Shadows Function EnumDeviceMonikers(<Out()> ByRef ppEnum As IEnumMoniker) As Integer

        <PreserveSig()> _
        Shadows Function get_DefaultPreferredComponentTypes(<Out()> ByRef ComponentTypes As IComponentTypes) As Integer

        <PreserveSig()> _
        Shadows Function put_DefaultPreferredComponentTypes(<[In]()> ByVal NewComponentTypes As IComponentTypes) As Integer

        <PreserveSig()> _
        Shadows Function get_FrequencyMapping(<Out(), MarshalAs(UnmanagedType.BStr)> ByRef pMapping As String) As Integer

        <PreserveSig()> _
        Shadows Function put_FrequencyMapping(<[In](), MarshalAs(UnmanagedType.BStr)> ByVal Mapping As String) As Integer

        <PreserveSig()> _
        Shadows Function get_DefaultLocator(<Out()> ByRef LocatorVal As ILocator) As Integer

        <PreserveSig()> _
        Shadows Function put_DefaultLocator(<[In]()> ByVal LocatorVal As ILocator) As Integer

        <PreserveSig()> _
        Shadows Function Clone(<Out()> ByRef NewTS As ITuningSpace) As Integer

        <PreserveSig()> _
        Function get_MinFrequency(<Out()> ByRef MinFrequencyVal As Integer) As Integer

        <PreserveSig()> _
        Function put_MinFrequency(<[In]()> ByVal NewMinFrequencyVal As Integer) As Integer

        <PreserveSig()> _
        Function get_MaxFrequency(<Out()> ByRef MaxFrequencyVal As Integer) As Integer

        <PreserveSig()> _
        Function put_MaxFrequency(<[In]()> ByVal NewMaxFrequencyVal As Integer) As Integer

        <PreserveSig()> _
        Function get_Step(<Out()> ByRef StepVal As Integer) As Integer

        <PreserveSig()> _
        Function put_Step(<[In]()> ByVal StepVal As Integer) As Integer
    End Interface

    <ComImport(), System.Security.SuppressUnmanagedCodeSecurity(), Guid("39DD45DA-2DA8-46BA-8A8A-87E2B73D983A"), InterfaceType(ComInterfaceType.InterfaceIsDual)> _
    Public Interface IAnalogRadioTuningSpace2
        Inherits IAnalogRadioTuningSpace

        <PreserveSig()> _
        Shadows Function get_UniqueName(<Out(), MarshalAs(UnmanagedType.BStr)> ByRef Name As String) As Integer

        <PreserveSig()> _
        Shadows Function put_UniqueName(<[In](), MarshalAs(UnmanagedType.BStr)> ByVal Name As String) As Integer

        <PreserveSig()> _
        Shadows Function get_FriendlyName(<Out(), MarshalAs(UnmanagedType.BStr)> ByRef Name As String) As Integer

        <PreserveSig()> _
        Shadows Function put_FriendlyName(<[In](), MarshalAs(UnmanagedType.BStr)> ByVal Name As String) As Integer

        <PreserveSig()> _
        Shadows Function get_CLSID(<Out(), MarshalAs(UnmanagedType.BStr)> ByRef SpaceCLSID As String) As Integer

        <PreserveSig()> _
        Shadows Function get_NetworkType(<Out(), MarshalAs(UnmanagedType.BStr)> ByRef NetworkTypeGuid As String) As Integer

        <PreserveSig()> _
        Shadows Function put_NetworkType(<[In](), MarshalAs(UnmanagedType.BStr)> ByVal NetworkTypeGuid As String) As Integer

        <PreserveSig()> _
        Shadows Function get__NetworkType(<Out()> ByRef NetworkTypeGuid As Guid) As Integer

        <PreserveSig()> _
        Shadows Function put__NetworkType(<[In](), MarshalAs(UnmanagedType.LPStruct)> ByVal NetworkTypeGuid As Guid) As Integer

        <PreserveSig()> _
        Shadows Function CreateTuneRequest(<Out()> ByRef TuneRequest As ITuneRequest) As Integer

        ' IEnumGUID**
        <PreserveSig()> _
        Shadows Function EnumCategoryGUIDs(<Out(), MarshalAs(UnmanagedType.IUnknown)> ByRef ppEnum As Object) As Integer

        <PreserveSig()> _
        Shadows Function EnumDeviceMonikers(<Out()> ByRef ppEnum As IEnumMoniker) As Integer

        <PreserveSig()> _
        Shadows Function get_DefaultPreferredComponentTypes(<Out()> ByRef ComponentTypes As IComponentTypes) As Integer

        <PreserveSig()> _
        Shadows Function put_DefaultPreferredComponentTypes(<[In]()> ByVal NewComponentTypes As IComponentTypes) As Integer

        <PreserveSig()> _
        Shadows Function get_FrequencyMapping(<Out(), MarshalAs(UnmanagedType.BStr)> ByRef pMapping As String) As Integer

        <PreserveSig()> _
        Shadows Function put_FrequencyMapping(<[In](), MarshalAs(UnmanagedType.BStr)> ByVal Mapping As String) As Integer

        <PreserveSig()> _
        Shadows Function get_DefaultLocator(<Out()> ByRef LocatorVal As ILocator) As Integer

        <PreserveSig()> _
        Shadows Function put_DefaultLocator(<[In]()> ByVal LocatorVal As ILocator) As Integer

        <PreserveSig()> _
        Shadows Function Clone(<Out()> ByRef NewTS As ITuningSpace) As Integer

        <PreserveSig()> _
        Shadows Function get_MinFrequency(<Out()> ByRef MinFrequencyVal As Integer) As Integer

        <PreserveSig()> _
        Shadows Function put_MinFrequency(<[In]()> ByVal NewMinFrequencyVal As Integer) As Integer

        <PreserveSig()> _
        Shadows Function get_MaxFrequency(<Out()> ByRef MaxFrequencyVal As Integer) As Integer

        <PreserveSig()> _
        Shadows Function put_MaxFrequency(<[In]()> ByVal NewMaxFrequencyVal As Integer) As Integer

        <PreserveSig()> _
        Shadows Function get_Step(<Out()> ByRef StepVal As Integer) As Integer

        <PreserveSig()> _
        Shadows Function put_Step(<[In]()> ByVal StepVal As Integer) As Integer

        Function get_CountryCode(ByRef CountryCodeVal As Integer) As Integer

        Function put_CountryCode(ByVal NewCountryCodeVal As Integer) As Integer
    End Interface

    <ComImport(), System.Security.SuppressUnmanagedCodeSecurity(), Guid("612AA885-66CF-4090-BA0A-566F5312E4CA"), InterfaceType(ComInterfaceType.InterfaceIsDual)> _
    Public Interface IATSCLocator2
        Inherits IATSCLocator

        <PreserveSig()> _
        Shadows Function get_CarrierFrequency(<Out()> ByRef Frequency As Integer) As Integer

        <PreserveSig()> _
        Shadows Function put_CarrierFrequency(<[In]()> ByVal Frequency As Integer) As Integer

        <PreserveSig()> _
        Shadows Function get_InnerFEC(<Out()> ByRef FEC As FECMethod) As Integer

        <PreserveSig()> _
        Shadows Function put_InnerFEC(<[In]()> ByVal FEC As FECMethod) As Integer

        <PreserveSig()> _
        Shadows Function get_InnerFECRate(<Out()> ByRef FEC As BinaryConvolutionCodeRate) As Integer

        <PreserveSig()> _
        Shadows Function put_InnerFECRate(<[In]()> ByVal FEC As BinaryConvolutionCodeRate) As Integer

        <PreserveSig()> _
        Shadows Function get_OuterFEC(<Out()> ByRef FEC As FECMethod) As Integer

        <PreserveSig()> _
        Shadows Function put_OuterFEC(<[In]()> ByVal FEC As FECMethod) As Integer

        <PreserveSig()> _
        Shadows Function get_OuterFECRate(<Out()> ByRef FEC As BinaryConvolutionCodeRate) As Integer

        <PreserveSig()> _
        Shadows Function put_OuterFECRate(<[In]()> ByVal FEC As BinaryConvolutionCodeRate) As Integer

        <PreserveSig()> _
        Shadows Function get_Modulation(<Out()> ByRef Modulation As ModulationType) As Integer

        <PreserveSig()> _
        Shadows Function put_Modulation(<[In]()> ByVal Modulation As ModulationType) As Integer

        <PreserveSig()> _
        Shadows Function get_SymbolRate(<Out()> ByRef Rate As Integer) As Integer

        <PreserveSig()> _
        Shadows Function put_SymbolRate(<[In]()> ByVal Rate As Integer) As Integer

        <PreserveSig()> _
        Shadows Function Clone(<Out()> ByRef NewLocator As ILocator) As Integer

        <PreserveSig()> _
        Shadows Function get_PhysicalChannel(<Out()> ByRef PhysicalChannel As Integer) As Integer

        <PreserveSig()> _
        Shadows Function put_PhysicalChannel(<[In]()> ByVal PhysicalChannel As Integer) As Integer

        <PreserveSig()> _
        Shadows Function get_TSID(<Out()> ByRef TSID As Integer) As Integer

        <PreserveSig()> _
        Shadows Function put_TSID(<[In]()> ByVal TSID As Integer) As Integer

        <PreserveSig()> _
        Function get_ProgramNumber(<Out()> ByRef ProgramNumber As Integer) As Integer

        <PreserveSig()> _
        Function put_ProgramNumber(<[In]()> ByVal ProgramNumber As Integer) As Integer
    End Interface

    <ComImport(), System.Security.SuppressUnmanagedCodeSecurity(), Guid("B10931ED-8BFE-4AB0-9DCE-E469C29A9729"), InterfaceType(ComInterfaceType.InterfaceIsDual)> _
    Public Interface IAuxInTuningSpace2
        Inherits IAuxInTuningSpace

        <PreserveSig()> _
        Shadows Function get_UniqueName(<Out(), MarshalAs(UnmanagedType.BStr)> ByRef Name As String) As Integer

        <PreserveSig()> _
        Shadows Function put_UniqueName(<[In](), MarshalAs(UnmanagedType.BStr)> ByVal Name As String) As Integer

        <PreserveSig()> _
        Shadows Function get_FriendlyName(<Out(), MarshalAs(UnmanagedType.BStr)> ByRef Name As String) As Integer

        <PreserveSig()> _
        Shadows Function put_FriendlyName(<[In](), MarshalAs(UnmanagedType.BStr)> ByVal Name As String) As Integer

        <PreserveSig()> _
        Shadows Function get_CLSID(<Out(), MarshalAs(UnmanagedType.BStr)> ByRef SpaceCLSID As String) As Integer

        <PreserveSig()> _
        Shadows Function get_NetworkType(<Out(), MarshalAs(UnmanagedType.BStr)> ByRef NetworkTypeGuid As String) As Integer

        <PreserveSig()> _
        Shadows Function put_NetworkType(<[In](), MarshalAs(UnmanagedType.BStr)> ByVal NetworkTypeGuid As String) As Integer

        <PreserveSig()> _
        Shadows Function get__NetworkType(<Out()> ByRef NetworkTypeGuid As Guid) As Integer

        <PreserveSig()> _
        Shadows Function put__NetworkType(<[In](), MarshalAs(UnmanagedType.LPStruct)> ByVal NetworkTypeGuid As Guid) As Integer

        <PreserveSig()> _
        Shadows Function CreateTuneRequest(<Out()> ByRef TuneRequest As ITuneRequest) As Integer

        ' IEnumGUID**
        <PreserveSig()> _
        Shadows Function EnumCategoryGUIDs(<Out(), MarshalAs(UnmanagedType.IUnknown)> ByRef ppEnum As Object) As Integer

        <PreserveSig()> _
        Shadows Function EnumDeviceMonikers(<Out()> ByRef ppEnum As IEnumMoniker) As Integer

        <PreserveSig()> _
        Shadows Function get_DefaultPreferredComponentTypes(<Out()> ByRef ComponentTypes As IComponentTypes) As Integer

        <PreserveSig()> _
        Shadows Function put_DefaultPreferredComponentTypes(<[In]()> ByVal NewComponentTypes As IComponentTypes) As Integer

        <PreserveSig()> _
        Shadows Function get_FrequencyMapping(<Out(), MarshalAs(UnmanagedType.BStr)> ByRef pMapping As String) As Integer

        <PreserveSig()> _
        Shadows Function put_FrequencyMapping(<[In](), MarshalAs(UnmanagedType.BStr)> ByVal Mapping As String) As Integer

        <PreserveSig()> _
        Shadows Function get_DefaultLocator(<Out()> ByRef LocatorVal As ILocator) As Integer

        <PreserveSig()> _
        Shadows Function put_DefaultLocator(<[In]()> ByVal LocatorVal As ILocator) As Integer

        <PreserveSig()> _
        Shadows Function Clone(<Out()> ByRef NewTS As ITuningSpace) As Integer

        Function get_CountryCode(ByRef CountryCodeVal As Integer) As Integer

        Function put_CountryCode(ByVal NewCountryCodeVal As Integer) As Integer
    End Interface

    <ComImport(), System.Security.SuppressUnmanagedCodeSecurity(), Guid("990237AE-AC11-4614-BE8F-DD217A4CB4CB"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)> _
    Public Interface IPersistTuneXmlUtility
        <PreserveSig()> _
        Function Deserialize(<[In]()> ByVal varValue As Object, <Out(), MarshalAs(UnmanagedType.IUnknown)> ByRef ppObject As Object) As Integer
    End Interface

    <ComImport(), System.Security.SuppressUnmanagedCodeSecurity(), Guid("992E165F-EA24-4b2f-9A1D-009D92120451"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)> _
    Public Interface IPersistTuneXmlUtility2
        Inherits IPersistTuneXmlUtility

        <PreserveSig()> _
        Shadows Function Deserialize(<[In]()> ByVal varValue As Object, <Out(), MarshalAs(UnmanagedType.IUnknown)> ByRef ppObject As Object) As Integer

        <PreserveSig()> _
        Function Serialize(ByVal piTuneRequest As ITuneRequest, <Out(), MarshalAs(UnmanagedType.BStr)> ByRef pString As String) As Integer
    End Interface

    <ComImport(), System.Security.SuppressUnmanagedCodeSecurity(), Guid("0754CD31-8D15-47A9-8215-D20064157244"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)> _
    Public Interface IPersistTuneXml
        Inherits IPersist

        <PreserveSig()> _
        Shadows Function GetClassID(ByRef pClassID As Guid) As Integer

        <PreserveSig()> _
        Function InitNew() As Integer

        <PreserveSig()> _
        Function Load(<[In]()> ByVal varValue As Object) As Integer

        <PreserveSig()> _
        Function Save(<Out()> ByRef pvarFragment As Object) As Integer
    End Interface

    <ComImport(), System.Security.SuppressUnmanagedCodeSecurity(), Guid("C0A4A1D4-2B3C-491A-BA22-499FBADD4D12"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)> _
    Public Interface IBDACreateTuneRequestEx
        <PreserveSig()> _
        Function CreateTuneRequestEx(<[In](), MarshalAs(UnmanagedType.LPStruct)> ByVal TuneRequestIID As Guid, <Out()> ByRef ppTuneRequest As ITuneRequest) As Integer
    End Interface

    <ComImport(), System.Security.SuppressUnmanagedCodeSecurity(), Guid("48F66A11-171A-419A-9525-BEEECD51584C"), InterfaceType(ComInterfaceType.InterfaceIsDual)> _
    Public Interface IDigitalCableLocator
        Inherits IATSCLocator2

        <PreserveSig()> _
        Shadows Function get_CarrierFrequency(<Out()> ByRef Frequency As Integer) As Integer

        <PreserveSig()> _
        Shadows Function put_CarrierFrequency(<[In]()> ByVal Frequency As Integer) As Integer

        <PreserveSig()> _
        Shadows Function get_InnerFEC(<Out()> ByRef FEC As FECMethod) As Integer

        <PreserveSig()> _
        Shadows Function put_InnerFEC(<[In]()> ByVal FEC As FECMethod) As Integer

        <PreserveSig()> _
        Shadows Function get_InnerFECRate(<Out()> ByRef FEC As BinaryConvolutionCodeRate) As Integer

        <PreserveSig()> _
        Shadows Function put_InnerFECRate(<[In]()> ByVal FEC As BinaryConvolutionCodeRate) As Integer

        <PreserveSig()> _
        Shadows Function get_OuterFEC(<Out()> ByRef FEC As FECMethod) As Integer

        <PreserveSig()> _
        Shadows Function put_OuterFEC(<[In]()> ByVal FEC As FECMethod) As Integer

        <PreserveSig()> _
        Shadows Function get_OuterFECRate(<Out()> ByRef FEC As BinaryConvolutionCodeRate) As Integer

        <PreserveSig()> _
        Shadows Function put_OuterFECRate(<[In]()> ByVal FEC As BinaryConvolutionCodeRate) As Integer

        <PreserveSig()> _
        Shadows Function get_Modulation(<Out()> ByRef Modulation As ModulationType) As Integer

        <PreserveSig()> _
        Shadows Function put_Modulation(<[In]()> ByVal Modulation As ModulationType) As Integer

        <PreserveSig()> _
        Shadows Function get_SymbolRate(<Out()> ByRef Rate As Integer) As Integer

        <PreserveSig()> _
        Shadows Function put_SymbolRate(<[In]()> ByVal Rate As Integer) As Integer

        <PreserveSig()> _
        Shadows Function Clone(<Out()> ByRef NewLocator As ILocator) As Integer

        <PreserveSig()> _
        Shadows Function get_PhysicalChannel(<Out()> ByRef PhysicalChannel As Integer) As Integer

        <PreserveSig()> _
        Shadows Function put_PhysicalChannel(<[In]()> ByVal PhysicalChannel As Integer) As Integer

        <PreserveSig()> _
        Shadows Function get_TSID(<Out()> ByRef TSID As Integer) As Integer

        <PreserveSig()> _
        Shadows Function put_TSID(<[In]()> ByVal TSID As Integer) As Integer

        <PreserveSig()> _
        Shadows Function get_ProgramNumber(<Out()> ByRef ProgramNumber As Integer) As Integer

        <PreserveSig()> _
        Shadows Function put_ProgramNumber(<[In]()> ByVal ProgramNumber As Integer) As Integer

    End Interface

    <ComImport(), System.Security.SuppressUnmanagedCodeSecurity(), Guid("156EFF60-86F4-4E28-89FC-109799FD57EE"), InterfaceType(ComInterfaceType.InterfaceIsDual)> _
    Public Interface IChannelIDTuneRequest
        Inherits ITuneRequest

        <PreserveSig()> _
        Shadows Function get_TuningSpace(<Out()> ByRef TuningSpace As ITuningSpace) As Integer

        <PreserveSig()> _
        Shadows Function get_Components(<Out()> ByRef Components As IComponents) As Integer

        <PreserveSig()> _
        Shadows Function Clone(<Out()> ByRef NewTuneRequest As ITuneRequest) As Integer

        <PreserveSig()> _
        Shadows Function get_Locator(<Out()> ByRef Locator As ILocator) As Integer

        <PreserveSig()> _
        Shadows Function put_Locator(<[In]()> ByVal Locator As ILocator) As Integer

        <PreserveSig()> _
        Function get_ChannelID(<Out(), MarshalAs(UnmanagedType.BStr)> ByRef ChannelID As String) As Integer

        <PreserveSig()> _
        Function put_ChannelID(<[In](), MarshalAs(UnmanagedType.BStr)> ByVal ChannelID As String) As Integer
    End Interface

    <ComImport(), System.Security.SuppressUnmanagedCodeSecurity(), Guid("34D1F26B-E339-430D-ABCE-738CB48984DC"), InterfaceType(ComInterfaceType.InterfaceIsDual)> _
    Public Interface IAnalogLocator
        Inherits ILocator

        <PreserveSig()> _
        Shadows Function get_CarrierFrequency(<Out()> ByRef Frequency As Integer) As Integer

        <PreserveSig()> _
        Shadows Function put_CarrierFrequency(<[In]()> ByVal Frequency As Integer) As Integer

        <PreserveSig()> _
        Shadows Function get_InnerFEC(<Out()> ByRef FEC As FECMethod) As Integer

        <PreserveSig()> _
        Shadows Function put_InnerFEC(<[In]()> ByVal FEC As FECMethod) As Integer

        <PreserveSig()> _
        Shadows Function get_InnerFECRate(<Out()> ByRef FEC As BinaryConvolutionCodeRate) As Integer

        <PreserveSig()> _
        Shadows Function put_InnerFECRate(<[In]()> ByVal FEC As BinaryConvolutionCodeRate) As Integer

        <PreserveSig()> _
        Shadows Function get_OuterFEC(<Out()> ByRef FEC As FECMethod) As Integer

        <PreserveSig()> _
        Shadows Function put_OuterFEC(<[In]()> ByVal FEC As FECMethod) As Integer

        <PreserveSig()> _
        Shadows Function get_OuterFECRate(<Out()> ByRef FEC As BinaryConvolutionCodeRate) As Integer

        <PreserveSig()> _
        Shadows Function put_OuterFECRate(<[In]()> ByVal FEC As BinaryConvolutionCodeRate) As Integer

        <PreserveSig()> _
        Shadows Function get_Modulation(<Out()> ByRef Modulation As ModulationType) As Integer

        <PreserveSig()> _
        Shadows Function put_Modulation(<[In]()> ByVal Modulation As ModulationType) As Integer

        <PreserveSig()> _
        Shadows Function get_SymbolRate(<Out()> ByRef Rate As Integer) As Integer

        <PreserveSig()> _
        Shadows Function put_SymbolRate(<[In]()> ByVal Rate As Integer) As Integer

        <PreserveSig()> _
        Shadows Function Clone(<Out()> ByRef NewLocator As ILocator) As Integer

        <PreserveSig()> _
        Function get_VideoStandard(<Out()> ByRef AVS As AnalogVideoStandard) As Integer

        <PreserveSig()> _
        Function put_VideoStandard(<[In]()> ByVal AVS As AnalogVideoStandard) As Integer
    End Interface

    <ComImport(), System.Security.SuppressUnmanagedCodeSecurity(), Guid("013F9F9C-B449-4ec7-A6D2-9D4F2FC70AE5"), InterfaceType(ComInterfaceType.InterfaceIsDual)> _
    Public Interface IDigitalCableTuningSpace
        Inherits IATSCTuningSpace

        <PreserveSig()> _
        Shadows Function get_UniqueName(<Out(), MarshalAs(UnmanagedType.BStr)> ByRef Name As String) As Integer

        <PreserveSig()> _
        Shadows Function put_UniqueName(<[In](), MarshalAs(UnmanagedType.BStr)> ByVal Name As String) As Integer

        <PreserveSig()> _
        Shadows Function get_FriendlyName(<Out(), MarshalAs(UnmanagedType.BStr)> ByRef Name As String) As Integer

        <PreserveSig()> _
        Shadows Function put_FriendlyName(<[In](), MarshalAs(UnmanagedType.BStr)> ByVal Name As String) As Integer

        <PreserveSig()> _
        Shadows Function get_CLSID(<Out(), MarshalAs(UnmanagedType.BStr)> ByRef SpaceCLSID As String) As Integer

        <PreserveSig()> _
        Shadows Function get_NetworkType(<Out(), MarshalAs(UnmanagedType.BStr)> ByRef NetworkTypeGuid As String) As Integer

        <PreserveSig()> _
        Shadows Function put_NetworkType(<[In](), MarshalAs(UnmanagedType.BStr)> ByVal NetworkTypeGuid As String) As Integer

        <PreserveSig()> _
        Shadows Function get__NetworkType(<Out()> ByRef NetworkTypeGuid As Guid) As Integer

        <PreserveSig()> _
        Shadows Function put__NetworkType(<[In](), MarshalAs(UnmanagedType.LPStruct)> ByVal NetworkTypeGuid As Guid) As Integer

        <PreserveSig()> _
        Shadows Function CreateTuneRequest(<Out()> ByRef TuneRequest As ITuneRequest) As Integer

        <PreserveSig()> _
        Shadows Function EnumCategoryGUIDs(<Out(), MarshalAs(UnmanagedType.IUnknown)> ByRef ppEnum As Object) As Integer
        ' IEnumGUID**
        <PreserveSig()> _
        Shadows Function EnumDeviceMonikers(<Out()> ByRef ppEnum As IEnumMoniker) As Integer

        <PreserveSig()> _
        Shadows Function get_DefaultPreferredComponentTypes(<Out()> ByRef ComponentTypes As IComponentTypes) As Integer

        <PreserveSig()> _
        Shadows Function put_DefaultPreferredComponentTypes(<[In]()> ByVal NewComponentTypes As IComponentTypes) As Integer

        <PreserveSig()> _
        Shadows Function get_FrequencyMapping(<Out(), MarshalAs(UnmanagedType.BStr)> ByRef pMapping As String) As Integer

        <PreserveSig()> _
        Shadows Function put_FrequencyMapping(<[In](), MarshalAs(UnmanagedType.BStr)> ByVal Mapping As String) As Integer

        <PreserveSig()> _
        Shadows Function get_DefaultLocator(<Out()> ByRef LocatorVal As ILocator) As Integer

        <PreserveSig()> _
        Shadows Function put_DefaultLocator(<[In]()> ByVal LocatorVal As ILocator) As Integer

        <PreserveSig()> _
        Shadows Function Clone(<Out()> ByRef NewTS As ITuningSpace) As Integer

        <PreserveSig()> _
        Shadows Function get_MinChannel(<Out()> ByRef MinChannelVal As Integer) As Integer

        <PreserveSig()> _
        Shadows Function put_MinChannel(<[In]()> ByVal NewMinChannelVal As Integer) As Integer

        <PreserveSig()> _
        Shadows Function get_MaxChannel(<Out()> ByRef MaxChannelVal As Integer) As Integer

        <PreserveSig()> _
        Shadows Function put_MaxChannel(<[In]()> ByVal NewMaxChannelVal As Integer) As Integer

        <PreserveSig()> _
        Shadows Function get_InputType(<Out()> ByRef InputTypeVal As TunerInputType) As Integer

        <PreserveSig()> _
        Shadows Function put_InputType(<[In]()> ByVal NewInputTypeVal As TunerInputType) As Integer

        <PreserveSig()> _
        Shadows Function get_CountryCode(<Out()> ByRef CountryCodeVal As Integer) As Integer

        <PreserveSig()> _
        Shadows Function put_CountryCode(<[In]()> ByVal NewCountryCodeVal As Integer) As Integer

        <PreserveSig()> _
        Shadows Function get_MinMinorChannel(<Out()> ByRef MinMinorChannelVal As Integer) As Integer

        <PreserveSig()> _
        Shadows Function put_MinMinorChannel(<[In]()> ByVal NewMinMinorChannelVal As Integer) As Integer

        <PreserveSig()> _
        Shadows Function get_MaxMinorChannel(<Out()> ByRef MaxMinorChannelVal As Integer) As Integer

        <PreserveSig()> _
        Shadows Function put_MaxMinorChannel(<[In]()> ByVal NewMaxMinorChannelVal As Integer) As Integer

        <PreserveSig()> _
        Shadows Function get_MinPhysicalChannel(<Out()> ByRef MinPhysicalChannelVal As Integer) As Integer

        <PreserveSig()> _
        Shadows Function put_MinPhysicalChannel(<[In]()> ByVal NewMinPhysicalChannelVal As Integer) As Integer

        <PreserveSig()> _
        Shadows Function get_MaxPhysicalChannel(<Out()> ByRef MaxPhysicalChannelVal As Integer) As Integer

        <PreserveSig()> _
        Shadows Function put_MaxPhysicalChannel(<[In]()> ByVal NewMaxPhysicalChannelVal As Integer) As Integer

        <PreserveSig()> _
        Function get_MinMajorChannel(<Out()> ByRef MinMajorChannelVal As Integer) As Integer

        <PreserveSig()> _
        Function put_MinMajorChannel(<[In]()> ByVal NewMinMajorChannelVal As Integer) As Integer

        <PreserveSig()> _
        Function get_MaxMajorChannel(<Out()> ByRef MaxMajorChannelVal As Integer) As Integer

        <PreserveSig()> _
        Function put_MaxMajorChannel(<[In]()> ByVal NewMaxMajorChannelVal As Integer) As Integer

        <PreserveSig()> _
        Function get_MinSourceID(<Out()> ByRef MinSourceIDVal As Integer) As Integer

        <PreserveSig()> _
        Function put_MinSourceID(<[In]()> ByVal NewMinSourceIDVal As Integer) As Integer

        <PreserveSig()> _
        Function get_MaxSourceID(<Out()> ByRef MaxSourceIDVal As Integer) As Integer

        <PreserveSig()> _
        Function put_MaxSourceID(<[In]()> ByVal NewMaxSourceIDVal As Integer) As Integer
    End Interface

    <ComImport(), System.Security.SuppressUnmanagedCodeSecurity(), Guid("2CFEB2A8-1787-4A24-A941-C6EAEC39C842"), InterfaceType(ComInterfaceType.InterfaceIsDual)> _
    Public Interface IAnalogAudioComponentType
        Inherits IComponentType

        <PreserveSig()> _
        Shadows Function get_Category(<Out()> ByRef Category As ComponentCategory) As Integer

        <PreserveSig()> _
        Shadows Function put_Category(<[In]()> ByVal Category As ComponentCategory) As Integer

        <PreserveSig()> _
        Shadows Function get_MediaMajorType(<Out(), MarshalAs(UnmanagedType.BStr)> ByRef MediaMajorType As String) As Integer

        <PreserveSig()> _
        Shadows Function put_MediaMajorType(<[In](), MarshalAs(UnmanagedType.BStr)> ByVal MediaMajorType As String) As Integer

        <PreserveSig()> _
        Shadows Function get__MediaMajorType(<Out()> ByRef MediaMajorType As Guid) As Integer

        <PreserveSig()> _
        Shadows Function put__MediaMajorType(<[In](), MarshalAs(UnmanagedType.LPStruct)> ByVal MediaMajorType As Guid) As Integer

        <PreserveSig()> _
        Shadows Function get_MediaSubType(<Out(), MarshalAs(UnmanagedType.BStr)> ByRef MediaSubType As String) As Integer

        <PreserveSig()> _
        Shadows Function put_MediaSubType(<[In](), MarshalAs(UnmanagedType.BStr)> ByVal MediaSubType As String) As Integer

        <PreserveSig()> _
        Shadows Function get__MediaSubType(<Out()> ByRef MediaSubType As Guid) As Integer

        <PreserveSig()> _
        Shadows Function put__MediaSubType(<[In](), MarshalAs(UnmanagedType.LPStruct)> ByVal MediaSubType As Guid) As Integer

        <PreserveSig()> _
        Shadows Function get_MediaFormatType(<Out(), MarshalAs(UnmanagedType.BStr)> ByRef MediaFormatType As String) As Integer

        <PreserveSig()> _
        Shadows Function put_MediaFormatType(<[In](), MarshalAs(UnmanagedType.BStr)> ByVal MediaFormatType As String) As Integer

        <PreserveSig()> _
        Shadows Function get__MediaFormatType(<Out()> ByRef MediaFormatType As Guid) As Integer

        <PreserveSig()> _
        Shadows Function put__MediaFormatType(<[In](), MarshalAs(UnmanagedType.LPStruct)> ByVal MediaFormatType As Guid) As Integer

        <PreserveSig()> _
        Shadows Function get_MediaType(<Out()> ByVal MediaType As AMMediaType) As Integer

        <PreserveSig()> _
        Shadows Function put_MediaType(<[In]()> ByVal MediaType As AMMediaType) As Integer

        <PreserveSig()> _
        Shadows Function Clone(<Out()> ByRef NewCT As IComponentType) As Integer

        <PreserveSig()> _
        Function get_AnalogAudioMode(<Out()> ByRef Mode As TVAudioMode) As Integer

        <PreserveSig()> _
        Function put_AnalogAudioMode(<[In]()> ByVal Mode As TVAudioMode) As Integer
    End Interface

End Namespace
