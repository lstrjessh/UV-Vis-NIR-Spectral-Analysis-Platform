
Imports System.Runtime.InteropServices

Namespace BDA

#If ALLOW_UNTESTED_INTERFACES Then

	Public Enum KSPropertyIPSink
		MulticastList
		AdapterDescription
		AdapterAddress
	End Enum

	<StructLayout(LayoutKind.Sequential)> _
	Public Structure PIDMap
		Public ulPID As Integer
		Public MediaSampleContent As MediaSampleContent
	End Structure

	Public Enum SmartCardStatusType
		CardInserted = 0
		CardRemoved
		CardError
		CardDataChanged
		CardFirmwareUpgrade
	End Enum

	Public Enum SmartCardAssociationType
		NotAssociated = 0
		Associated
		AssociationUnknown
	End Enum

	Public Enum LocationCodeSchemeType
		SCTE_18 = 0
	End Enum

	<StructLayout(LayoutKind.Sequential, Pack := 1)> _
	Public Structure EALocationCodeType
		Public LocationCodeScheme As LocationCodeSchemeType
		Public StateCode As Byte
		Public CountySubdivision As Byte
		Public CountyCode As Short
	End Structure

	Public Enum EntitlementType
		Entitled = 0
		NotEntitled
		TechnicalFailure
	End Enum

	Public Enum UICloseReasonType
		NotReady = 0
		UserClosed
		SystemClosed
		DeviceClosed
		ErrorClosed
	End Enum

	<StructLayout(LayoutKind.Sequential)> _
	Public Structure SmartCardApplication
		Public ApplicationType As ApplicationTypeType
		Public ApplicationVersion As Short
		<MarshalAs(UnmanagedType.BStr)> _
		Public pbstrApplicationName As String
		<MarshalAs(UnmanagedType.BStr)> _
		Public pbstrApplicationURL As String
	End Structure

	Public Enum BDA_DrmPairingError
		Succeeded = 0
		HardwareFailure
		NeedRevocationData
		NeedIndiv
		Other
		DrmInitFailed
		DrmNotPaired
		DrmRePairSoon
		Aborted
		NeedSDKUpdate
	End Enum

	Public Enum BDA_CONDITIONALACCESS_REQUESTTYPE
		Unspecified = 0
		NotPossible
		Possible
		PossibleNoStreamingDisruption
	End Enum

	Public Enum BDA_CONDITIONALACCESS_MMICLOSEREASON
		Unspecified = 0
		ClosedItself
		TunerRequestedClose
		DialogTimeout
		DialogFocusChange
		DialogUserDismissed
		DialogUserNotAvailable
	End Enum

	Public Enum MUX_PID_TYPE
		Other = -1
		ElementaryStream
		MPEG2SectionPSISI
	End Enum

	Public Class BDA_MUX_PIDLISTITEM
		Public usPIDNumber As Short
		Public usProgramNumber As Short
		Public ePIDType As MUX_PID_TYPE
	End Class

	Public Class BDA_SIGNAL_TIMEOUTS
		Public ulCarrierTimeoutMs As Integer
		Public ulScanningTimeoutMs As Integer
		Public ulTuningTimeoutMs As Integer
	End Class

#End If

    Public Enum BDAChangeState
        ChangesComplete = 0
        ChangesPending
    End Enum

    Public Enum MulticastMode
        PromiscuousMulticast = 0
        FilteredMulticast
        NoMulticast
    End Enum

    Public Enum MediaSampleContent
        TransportPacket
        ElementaryStream
        Mpeg2PSI
        TransportPayload
    End Enum

    <StructLayout(LayoutKind.Sequential)> _
    Public Structure BDANodeDescriptor
        Public ulBdaNodeType As Integer
        Public guidFunction As Guid
        Public guidName As Guid
    End Structure

    <Flags()> _
    Public Enum CCSubstreamService
        None = 0
        CC1 = &H1       'CC1 (caption channel) 
        CC2 = &H2       'CC2 (caption channel) 
        T1 = &H4        ' T1 (text channel) 
        T2 = &H8        ' T2 (text channel) 
        CC3 = &H100     ' CC3 (caption channel) 
        CC4 = &H200     ' CC4 (caption channel) 
        T3 = &H400      ' T3 (text channel) 
        T4 = &H800      ' T4 (text channel) 
        XDS = &H1000    ' Extended Data Services (XDS) 
        Field1 = &HF    ' Bitmask to filter field 1 substreams. 
        Field2 = &H1F00 'Bitmask to filter field 2 substreams 
    End Enum


#If ALLOW_UNTESTED_INTERFACES Then

	<ComImport, System.Security.SuppressUnmanagedCodeSecurity, Guid("fd501041-8ebe-11ce-8183-00aa00577da2"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)> _
	Public Interface IBDA_NetworkProvider
		<PreserveSig> _
		Function PutSignalSource(<[In]> ulSignalSource As Integer) As Integer

		<PreserveSig> _
		Function GetSignalSource(<Out> ByRef pulSignalSource As Integer) As Integer

		<PreserveSig> _
		Function GetNetworkType(<Out> ByRef pguidNetworkType As Guid) As Integer

		<PreserveSig> _
		Function PutTuningSpace(<[In], MarshalAs(UnmanagedType.LPStruct)> guidTuningSpace As Guid) As Integer

		<PreserveSig> _
		Function GetTuningSpace(<Out> ByRef pguidTuingSpace As Guid) As Integer

		<PreserveSig> _
		Function RegisterDeviceFilter(<[In], MarshalAs(UnmanagedType.[Interface])> pUnkFilterControl As Object, <Out> ByRef ppvRegisitrationContext As Integer) As Integer

		<PreserveSig> _
		Function UnRegisterDeviceFilter(<[In]> pvRegistrationContext As Integer) As Integer
	End Interface

	<ComImport, System.Security.SuppressUnmanagedCodeSecurity, Guid("71985F46-1CA1-11d3-9CC8-00C04F7971E0"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)> _
	Public Interface IBDA_VoidTransform
		<PreserveSig> _
		Function Start() As Integer

		<PreserveSig> _
		Function [Stop]() As Integer
	End Interface

	<ComImport, System.Security.SuppressUnmanagedCodeSecurity, Guid("DDF15B0D-BD25-11d2-9CA0-00C04F7971E0"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)> _
	Public Interface IBDA_NullTransform
		<PreserveSig> _
		Function Start() As Integer

		<PreserveSig> _
		Function [Stop]() As Integer
	End Interface

	<ComImport, System.Security.SuppressUnmanagedCodeSecurity, Guid("afb6c2a2-2c41-11d3-8a60-0000f81e0e4a"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)> _
	Public Interface IEnumPIDMap
		<PreserveSig> _
		Function [Next](<[In]> cRequest As Integer, <Out, MarshalAs(UnmanagedType.LPArray, SizeParamIndex := 0, ArraySubType := UnmanagedType.Struct)> pPIDMap As PIDMap(), <[In]> pcReceived As IntPtr) As Integer

		<PreserveSig> _
		Function Skip(<[In]> cRecords As Integer) As Integer

		<PreserveSig> _
		Function Reset() As Integer

		<PreserveSig> _
		Function Clone(<Out> ByRef ppIEnumPIDMap As IEnumPIDMap) As Integer
	End Interface

	<ComImport, System.Security.SuppressUnmanagedCodeSecurity, Guid("0DED49D5-A8B7-4d5d-97A1-12B0C195874D"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)> _
	Public Interface IBDA_PinControl
		<PreserveSig> _
		Function GetPinID(<Out> ByRef pulPinID As Integer) As Integer

		<PreserveSig> _
		Function GetPinType(<Out> ByRef pulPinType As Integer) As Integer

		<PreserveSig> _
		Function RegistrationContext(<Out> ByRef pulRegistrationCtx As Integer) As Integer
	End Interface

	<ComImport, System.Security.SuppressUnmanagedCodeSecurity, Guid("34518D13-1182-48e6-B28F-B24987787326"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)> _
	Public Interface IBDA_AutoDemodulateEx
		Inherits IBDA_AutoDemodulate

		<PreserveSig> _
		Shadows Function put_AutoDemodulate() As Integer

		<PreserveSig> _
		Function get_SupportedDeviceNodeTypes(<[In]> ulcDeviceNodeTypesMax As Integer, <Out> ByRef pulcDeviceNodeTypes As Integer, <[In], Out, MarshalAs(UnmanagedType.LPArray, ArraySubType := UnmanagedType.Struct)> pguidDeviceNodeTypes As Guid()) As Integer

		<PreserveSig> _
		Function get_SupportedVideoFormats(<Out> ByRef pulAMTunerModeType As AMTunerModeType, <Out> ByRef pulAnalogVideoStandard As AnalogVideoStandard) As Integer

		<PreserveSig> _
		Function get_AuxInputCount(<Out> ByRef pulCompositeCount As Integer, <Out> ByRef pulSvideoCount As Integer) As Integer
	End Interface

	<ComImport, System.Security.SuppressUnmanagedCodeSecurity, Guid("D806973D-3EBE-46de-8FBB-6358FE784208"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)> _
	Public Interface IBDA_EasMessage
		<PreserveSig> _
		Function get_EasMessage(<[In]> ulEventID As Integer, <Out, MarshalAs(UnmanagedType.IUnknown)> ByRef ppEASObject As Object) As Integer
	End Interface

	<ComImport, System.Security.SuppressUnmanagedCodeSecurity, Guid("8E882535-5F86-47AB-86CF-C281A72A0549"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)> _
	Public Interface IBDA_TransportStreamInfo
		<PreserveSig> _
		Function get_PatTableTickCount(<Out> ByRef pPatTickCount As Integer) As Integer
	End Interface

	<ComImport, System.Security.SuppressUnmanagedCodeSecurity, Guid("CD51F1E0-7BE9-4123-8482-A2A796C0A6B0"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)> _
	Public Interface IBDA_ConditionalAccess
		<PreserveSig> _
		Function get_SmartCardStatus(<Out> ByRef pCardStatus As SmartCardStatusType, <Out> ByRef pCardAssociation As SmartCardAssociationType, <Out, MarshalAs(UnmanagedType.BStr)> ByRef pbstrCardError As String, <Out, MarshalAs(UnmanagedType.VariantBool)> ByRef pfOOBLocked As Boolean) As Integer

		<PreserveSig> _
		Function get_SmartCardInfo(<Out, MarshalAs(UnmanagedType.BStr)> ByRef pbstrCardName As String, <Out, MarshalAs(UnmanagedType.BStr)> ByRef pbstrCardManufacturer As String, <Out, MarshalAs(UnmanagedType.VariantBool)> ByRef pfDaylightSavings As Boolean, <Out> ByRef pbyRatingRegion As Byte, <Out> ByRef plTimeZoneOffsetMinutes As Integer, <Out, MarshalAs(UnmanagedType.BStr)> ByRef pbstrLanguage As String, _
			<Out> ByRef pEALocationCode As EALocationCodeType) As Integer

		<PreserveSig> _
		Function get_SmartCardApplications(<[In], Out> ByRef pulcApplications As Integer, <[In]> ulcApplicationsMax As Integer, <[In], Out> rgApplications As SmartCardApplication()) As Integer

		<PreserveSig> _
		Function get_Entitlement(<[In]> usVirtualChannel As Short, <Out> ByRef pEntitlement As EntitlementType) As Integer

		<PreserveSig> _
		Function TuneByChannel(<[In]> usVirtualChannel As Short) As Integer

		<PreserveSig> _
		Function SetProgram(<[In]> usProgramNumber As Short) As Integer

		<PreserveSig> _
		Function AddProgram(<[In]> usProgramNumber As Short) As Integer

		<PreserveSig> _
		Function RemoveProgram(<[In]> usProgramNumber As Short) As Integer

		<PreserveSig> _
		Function GetModuleUI(<[In]> byDialogNumber As Byte, <Out, MarshalAs(UnmanagedType.BStr)> ByRef pbstrURL As String) As Integer

		<PreserveSig> _
		Function InformUIClosed(<[In]> byDialogNumber As Byte, <[In]> CloseReason As UICloseReasonType) As Integer
	End Interface

	<ComImport, System.Security.SuppressUnmanagedCodeSecurity, Guid("20e80cb5-c543-4c1b-8eb3-49e719eee7d4"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)> _
	Public Interface IBDA_DiagnosticProperties
		Inherits IPropertyBag
	End Interface

	<ComImport, System.Security.SuppressUnmanagedCodeSecurity, Guid("F98D88B0-1992-4cd6-A6D9-B9AFAB99330D"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)> _
	Public Interface IBDA_DRM
		<PreserveSig> _
		Function GetDRMPairingStatus(<Out> ByRef pdwStatus As BDA_DrmPairingError, <Out> ByRef phError As Integer) As Integer

		<PreserveSig> _
		Function PerformDRMPairing(<[In], MarshalAs(UnmanagedType.Bool)> fSync As Boolean) As Integer
	End Interface

	<ComImport, System.Security.SuppressUnmanagedCodeSecurity, Guid("7DEF4C09-6E66-4567-A819-F0E17F4A81AB"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)> _
	Public Interface IBDA_AUX
		<PreserveSig> _
		Function QueryCapabilities(ByRef pdwNumAuxInputsBSTR As Integer) As Integer

		<PreserveSig> _
		Function EnumCapability(<[In]> dwIndex As Integer, ByRef dwInputID As Integer, ByRef pConnectorType As Guid, ByRef ConnTypeNum As Integer, ByRef NumVideoStds As Integer, ByRef AnalogStds As Long) As Integer
	End Interface

	<ComImport, System.Security.SuppressUnmanagedCodeSecurity, Guid("497C3418-23CB-44BA-BB62-769F506FCEA7"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)> _
	Public Interface IBDA_ConditionalAccessEx
		<PreserveSig> _
		Function CheckEntitlementToken(<[In]> ulDialogRequest As Integer, <[In], MarshalAs(UnmanagedType.BStr)> bstrLanguage As String, <[In]> RequestType As BDA_CONDITIONALACCESS_REQUESTTYPE, <[In]> ulcbEntitlementTokenLen As Integer, <[In]> ByRef pbEntitlementToken As Byte, ByRef pulDescrambleStatus As Integer) As Integer

		<PreserveSig> _
		Function SetCaptureToken(<[In]> ulcbCaptureTokenLen As Integer, <[In]> ByRef pbCaptureToken As Byte) As Integer

		<PreserveSig> _
		Function OpenBroadcastMmi(<[In]> ulDialogRequest As Integer, <[In], MarshalAs(UnmanagedType.BStr)> bstrLanguage As String, <[In]> EventId As Integer) As Integer

		<PreserveSig> _
		Function CloseMmiDialog(<[In]> ulDialogRequest As Integer, <[In], MarshalAs(UnmanagedType.BStr)> bstrLanguage As String, <[In]> ulDialogNumber As Integer, <[In]> ReasonCode As BDA_CONDITIONALACCESS_MMICLOSEREASON, ByRef pulSessionResult As Integer) As Integer

		<PreserveSig> _
		Function CreateDialogRequestNumber(ByRef pulDialogRequestNumber As Integer) As Integer
	End Interface

	<ComImport, System.Security.SuppressUnmanagedCodeSecurity, InterfaceType(ComInterfaceType.InterfaceIsIUnknown), Guid("525ED3EE-5CF3-4E1E-9A06-5368A84F9A6E")> _
	Public Interface IBDA_DigitalDemodulator2
		Inherits IBDA_DigitalDemodulator

		<PreserveSig> _
		Shadows Function put_ModulationType(<[In]> ByRef pModulationType As ModulationType) As Integer

		<PreserveSig> _
		Shadows Function get_ModulationType(<Out> ByRef pModulationType As ModulationType) As Integer

		<PreserveSig> _
		Shadows Function put_InnerFECMethod(<[In]> ByRef pFECMethod As FECMethod) As Integer

		<PreserveSig> _
		Shadows Function get_InnerFECMethod(<Out> ByRef pFECMethod As FECMethod) As Integer

		<PreserveSig> _
		Shadows Function put_InnerFECRate(<[In]> ByRef pFECRate As BinaryConvolutionCodeRate) As Integer

		<PreserveSig> _
		Shadows Function get_InnerFECRate(<Out> ByRef pFECRate As BinaryConvolutionCodeRate) As Integer

		<PreserveSig> _
		Shadows Function put_OuterFECMethod(<[In]> ByRef pFECMethod As FECMethod) As Integer

		<PreserveSig> _
		Shadows Function get_OuterFECMethod(<Out> ByRef pFECMethod As FECMethod) As Integer

		<PreserveSig> _
		Shadows Function put_OuterFECRate(<[In]> ByRef pFECRate As BinaryConvolutionCodeRate) As Integer

		<PreserveSig> _
		Shadows Function get_OuterFECRate(<Out> ByRef pFECRate As BinaryConvolutionCodeRate) As Integer

		<PreserveSig> _
		Shadows Function put_SymbolRate(<[In]> ByRef pSymbolRate As Integer) As Integer

		<PreserveSig> _
		Shadows Function get_SymbolRate(<Out> ByRef pSymbolRate As Integer) As Integer

		<PreserveSig> _
		Shadows Function put_SpectralInversion(<[In]> ByRef pSpectralInversion As SpectralInversion) As Integer

		<PreserveSig> _
		Shadows Function get_SpectralInversion(<Out> ByRef pSpectralInversion As SpectralInversion) As Integer

		<PreserveSig> _
		Function put_GuardInterval(<[In]> ByRef pGuardInterval As GuardInterval) As Integer

		<PreserveSig> _
		Function get_GuardInterval(<[In], Out> ByRef pGuardInterval As GuardInterval) As Integer

		<PreserveSig> _
		Function put_TransmissionMode(<[In]> ByRef pTransmissionMode As TransmissionMode) As Integer

		<PreserveSig> _
		Function get_TransmissionMode(<[In], Out> ByRef pTransmissionMode As TransmissionMode) As Integer

		<PreserveSig> _
		Function put_RollOff(<[In]> ByRef pRollOff As RollOff) As Integer

		<PreserveSig> _
		Function get_RollOff(<[In], Out> ByRef pRollOff As RollOff) As Integer

		<PreserveSig> _
		Function put_Pilot(<[In]> ByRef pPilot As Pilot) As Integer

		<PreserveSig> _
		Function get_Pilot(<[In], Out> ByRef pPilot As Pilot) As Integer
	End Interface

	<ComImport, System.Security.SuppressUnmanagedCodeSecurity, Guid("F84E2AB0-3C6B-45E3-A0FC-8669D4B81F11"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)> _
	Public Interface IBDA_DiseqCommand
		<PreserveSig> _
		Function put_EnableDiseqCommands(<[In]> bEnable As Byte) As Integer

		<PreserveSig> _
		Function put_DiseqLNBSource(<[In]> ulLNBSource As Integer) As Integer

		<PreserveSig> _
		Function put_DiseqUseToneBurst(<[In]> bUseToneBurst As Byte) As Integer

		<PreserveSig> _
		Function put_DiseqRepeats(<[In]> ulRepeats As Integer) As Integer

		<PreserveSig> _
		Function put_DiseqSendCommand(<[In]> ulRequestId As Integer, <[In]> ulcbCommandLen As Integer, <[In]> ByRef pbCommand As Byte) As Integer

		<PreserveSig> _
		Function get_DiseqResponse(<[In]> ulRequestId As Integer, <[In], Out> ByRef pulcbResponseLen As Integer, <[In], Out> ByRef pbResponse As Byte) As Integer
	End Interface

	<ComImport, System.Security.SuppressUnmanagedCodeSecurity, Guid("1F9BC2A5-44A3-4C52-AAB1-0BBCE5A1381D"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)> _
	Public Interface IBDA_DRIDRMService
		<PreserveSig> _
		Function SetDRM(<[In], MarshalAs(UnmanagedType.BStr)> bstrNewDrm As String) As Integer

		<PreserveSig> _
		Function GetDRMStatus(<MarshalAs(UnmanagedType.BStr)> ByRef pbstrDrmUuidList As String, ByRef DrmUuid As Guid) As Integer

		<PreserveSig> _
		Function GetPairingStatus(<[In], Out> ByRef penumPairingStatus As BDA_DrmPairingError) As Integer
	End Interface

	<ComImport, System.Security.SuppressUnmanagedCodeSecurity, Guid("05C690F8-56DB-4BB2-B053-79C12098BB26"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)> _
	Public Interface IBDA_DRIWMDRMSession
		<PreserveSig> _
		Function AcknowledgeLicense(<[In], MarshalAs(UnmanagedType.[Error])> hrLicenseAck As Integer) As Integer

		<PreserveSig> _
		Function ProcessLicenseChallenge(<[In]> dwcbLicenseMessage As Integer, <[In]> ByRef pbLicenseMessage As Byte, <[In], Out> ByRef pdwcbLicenseResponse As Integer, <[In], Out> ppbLicenseResponse As IntPtr) As Integer

		<PreserveSig> _
		Function ProcessRegistrationChallenge(<[In]> dwcbRegistrationMessage As Integer, <[In]> ByRef pbRegistrationMessage As Byte, <[In], Out> ByRef pdwcbRegistrationResponse As Integer, <[In], Out> ppbRegistrationResponse As IntPtr) As Integer

		<PreserveSig> _
		Function SetRevInfo(<[In]> dwRevInfoLen As Integer, <[In]> ByRef pbRevInfo As Byte, <[In], Out> ByRef pdwResponse As Integer) As Integer

		<PreserveSig> _
		Function SetCrl(<[In]> dwCrlLen As Integer, <[In]> ByRef pbCrlLen As Byte, <[In], Out> ByRef pdwResponse As Integer) As Integer

		<PreserveSig> _
		Function GetHMSAssociationData() As Integer

		<PreserveSig> _
		Function GetLastCardeaError(<[In], Out> ByRef pdwError As Integer) As Integer
	End Interface

	<ComImport, System.Security.SuppressUnmanagedCodeSecurity, Guid("BFF6B5BB-B0AE-484C-9DCA-73528FB0B46E"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)> _
	Public Interface IBDA_DRMService
		<PreserveSig> _
		Function SetDRM(<[In]> ByRef puuidNewDrm As Guid) As Integer

		<PreserveSig> _
		Function GetDRMStatus(<MarshalAs(UnmanagedType.BStr)> ByRef pbstrDrmUuidList As String, ByRef DrmUuid As Guid) As Integer
	End Interface

	<ComImport, System.Security.SuppressUnmanagedCodeSecurity, Guid("3A8BAD59-59FE-4559-A0BA-396CFAA98AE3"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)> _
	Public Interface IBDA_Encoder
		<PreserveSig> _
		Function QueryCapabilities(ByRef NumAudioFmts As Integer, ByRef NumVideoFmts As Integer) As Integer

		<PreserveSig> _
		Function EnumAudioCapability(<[In]> FmtIndex As Integer, ByRef MethodID As Integer, ByRef AlgorithmType As Integer, ByRef SamplingRate As Integer, ByRef BitDepth As Integer, ByRef NumChannels As Integer) As Integer

		<PreserveSig> _
		Function EnumVideoCapability(<[In]> FmtIndex As Integer, ByRef MethodID As Integer, ByRef AlgorithmType As Integer, ByRef VerticalSize As Integer, ByRef HorizontalSize As Integer, ByRef AspectRatio As Integer, _
			ByRef FrameRateCode As Integer, ByRef ProgressiveSequence As Integer) As Integer

		<PreserveSig> _
		Function SetParameters(<[In]> AudioBitrateMode As Integer, <[In]> AudioBitrate As Integer, <[In]> AudioMethodID As Integer, <[In]> AudioProgram As Integer, <[In]> VideoBitrateMode As Integer, <[In]> VideoBitrate As Integer, _
			<[In]> VideoMethodID As Integer) As Integer

		<PreserveSig> _
		Function GetState(ByRef AudioBitrateMax As Integer, ByRef AudioBitrateMin As Integer, ByRef AudioBitrateMode As Integer, ByRef AudioBitrateStepping As Integer, ByRef AudioBitrate As Integer, ByRef AudioMethodID As Integer, _
			ByRef AvailableAudioPrograms As Integer, ByRef AudioProgram As Integer, ByRef VideoBitrateMax As Integer, ByRef VideoBitrateMin As Integer, ByRef VideoBitrateMode As Integer, ByRef VideoBitrate As Integer, _
			ByRef VideoBitrateStepping As Integer, ByRef VideoMethodID As Integer, ByRef SignalSourceID As Integer, ByRef SignalFormat As Long, ByRef SignalLock As Integer, ByRef SignalLevel As Integer, _
			ByRef SignalToNoiseRatio As Integer) As Integer
	End Interface

	<ComImport, System.Security.SuppressUnmanagedCodeSecurity, Guid("207C413F-00DC-4C61-BAD6-6FEE1FF07064"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)> _
	Public Interface IBDA_EventingService
		<PreserveSig> _
		Function CompleteEvent(<[In]> ulEventID As Integer, <[In]> ulEventResult As Integer) As Integer
	End Interface

	<ComImport, System.Security.SuppressUnmanagedCodeSecurity, Guid("138ADC7E-58AE-437F-B0B4-C9FE19D5B4AC"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)> _
	Public Interface IBDA_FDC
		<PreserveSig> _
		Function GetStatus(ByRef CurrentBitrate As Integer, ByRef CarrierLock As Integer, ByRef CurrentFrequency As Integer, ByRef CurrentSpectrumInversion As Integer, <MarshalAs(UnmanagedType.BStr)> ByRef CurrentPIDList As String, <MarshalAs(UnmanagedType.BStr)> ByRef CurrentTIDList As String, _
			ByRef Overflow As Integer) As Integer

		<PreserveSig> _
		Function RequestTables(<[In], MarshalAs(UnmanagedType.BStr)> TableIDs As String) As Integer

		<PreserveSig> _
		Function AddPid(<[In], MarshalAs(UnmanagedType.BStr)> PidsToAdd As String, ByRef RemainingFilterEntries As Integer) As Integer

		<PreserveSig> _
		Function RemovePid(<[In], MarshalAs(UnmanagedType.BStr)> PidsToRemove As String) As Integer

		<PreserveSig> _
		Function AddTid(<[In], MarshalAs(UnmanagedType.BStr)> TidsToAdd As String, <MarshalAs(UnmanagedType.BStr)> ByRef CurrentTIDList As String) As Integer

		<PreserveSig> _
		Function RemoveTid(<[In], MarshalAs(UnmanagedType.BStr)> TidsToRemove As String) As Integer

		<PreserveSig> _
		Function GetTableSection(ByRef Pid As Integer, <[In]> MaxBufferSize As Integer, ByRef ActualSize As Integer, ByRef SecBuffer As Byte) As Integer
	End Interface

	<ComImport, System.Security.SuppressUnmanagedCodeSecurity, Guid("C0AFCB73-23E7-4BC6-BAFA-FDC167B4719F"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)> _
	Public Interface IBDA_GuideDataDeliveryService
		<PreserveSig> _
		Function GetGuideDataType(ByRef pguidDataType As Guid) As Integer

		<PreserveSig> _
		Function GetGuideData(<[In], Out> ByRef pulcbBufferLen As Integer, ByRef pbBuffer As Byte, ByRef pulGuideDataPercentageProgress As Integer) As Integer

		<PreserveSig> _
		Function RequestGuideDataUpdate() As Integer

		<PreserveSig> _
		Function GetTuneXmlFromServiceIdx(<[In]> ul64ServiceIdx As Long, <MarshalAs(UnmanagedType.BStr)> ByRef pbstrTuneXml As String) As Integer

		<PreserveSig> _
		Function GetServices(<[In], Out> ByRef pulcbBufferLen As Integer, ByRef pbBuffer As Byte) As Integer

		<PreserveSig> _
		Function GetServiceInfoFromTuneXml(<[In], MarshalAs(UnmanagedType.BStr)> bstrTuneXml As String, <MarshalAs(UnmanagedType.BStr)> ByRef pbstrServiceDescription As String) As Integer
	End Interface

	<ComImport, System.Security.SuppressUnmanagedCodeSecurity, Guid("5E68C627-16C2-4E6C-B1E2-D00170CDAA0F"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)> _
	Public Interface IBDA_ISDBConditionalAccess
		<PreserveSig> _
		Function SetIsdbCasRequest(<[In]> ulRequestId As Integer, <[In]> ulcbRequestBufferLen As Integer, <[In]> ByRef pbRequestBuffer As Byte) As Integer
	End Interface

	<ComImport, System.Security.SuppressUnmanagedCodeSecurity, Guid("942AAFEC-4C05-4C74-B8EB-8706C2A4943F"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)> _
	Public Interface IBDA_MUX
		<PreserveSig> _
		Function SetPidList(<[In]> ulPidListCount As Integer, <[In]> ByRef pbPidListBuffer As BDA_MUX_PIDLISTITEM) As Integer

		<PreserveSig> _
		Function GetPidList(<[In], Out> ByRef pulPidListCount As Integer, <[In], Out> ByRef pbPidListBuffer As BDA_MUX_PIDLISTITEM) As Integer
	End Interface

	<ComImport, System.Security.SuppressUnmanagedCodeSecurity, Guid("7F0B3150-7B81-4AD4-98E3-7E9097094301"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)> _
	Public Interface IBDA_NameValueService
		<PreserveSig> _
		Function GetValueNameByIndex(<[In]> ulIndex As Integer, <MarshalAs(UnmanagedType.BStr)> ByRef pbstrName As String) As Integer

		<PreserveSig> _
		Function GetValue(<[In], MarshalAs(UnmanagedType.BStr)> bstrName As String, <[In], MarshalAs(UnmanagedType.BStr)> bstrLanguage As String, <MarshalAs(UnmanagedType.BStr)> ByRef pbstrValue As String) As Integer

		<PreserveSig> _
		Function SetValue(<[In]> ulDialogRequest As Integer, <[In], MarshalAs(UnmanagedType.BStr)> bstrLanguage As String, <[In], MarshalAs(UnmanagedType.BStr)> bstrName As String, <[In], MarshalAs(UnmanagedType.BStr)> bstrValue As String, <[In]> ulReserved As Integer) As Integer
	End Interface

	<ComImport, System.Security.SuppressUnmanagedCodeSecurity, Guid("1DCFAFE9-B45E-41B3-BB2A-561EB129AE98"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)> _
	Public Interface IBDA_TransportStreamSelector
		<PreserveSig> _
		Function SetTSID(<[In]> usTSID As Short) As Integer

		<PreserveSig> _
		Function GetTSInformation(<[In], Out> ByRef pulTSInformationBufferLen As Integer, ByRef pbTSInformationBuffer As Byte) As Integer
	End Interface

	<ComImport, System.Security.SuppressUnmanagedCodeSecurity, Guid("53B14189-E478-4B7A-A1FF-506DB4B99DFE"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)> _
	Public Interface IBDA_UserActivityService
		<PreserveSig> _
		Function SetCurrentTunerUseReason(<[In]> dwUseReason As Integer) As Integer

		<PreserveSig> _
		Function GetUserActivityInterval(ByRef pdwActivityInterval As Integer) As Integer

		<PreserveSig> _
		Function UserActivityDetected() As Integer
	End Interface

	<ComImport, System.Security.SuppressUnmanagedCodeSecurity, Guid("4BE6FA3D-07CD-4139-8B80-8C18BA3AEC88"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)> _
	Public Interface IBDA_WMDRMSession
		<PreserveSig> _
		Function GetStatus(ByRef MaxCaptureToken As Integer, ByRef MaxStreamingPid As Integer, ByRef MaxLicense As Integer, ByRef MinSecurityLevel As Integer, ByRef RevInfoSequenceNumber As Integer, ByRef RevInfoIssuedTime As Long, _
			ByRef RevInfoTTL As Integer, ByRef RevListVersion As Integer, ByRef ulState As Integer) As Integer

		<PreserveSig> _
		Function SetRevInfo(<[In]> ulRevInfoLen As Integer, <[In]> ByRef pbRevInfo As Byte) As Integer

		<PreserveSig> _
		Function SetCrl(<[In]> ulCrlLen As Integer, <[In]> ByRef pbCrlLen As Byte) As Integer

		<PreserveSig> _
		Function TransactMessage(<[In]> ulcbRequest As Integer, <[In]> ByRef pbRequest As Byte, <[In], Out> ByRef pulcbResponse As Integer, <[In], Out> ByRef pbResponse As Byte) As Integer

		<PreserveSig> _
		Function GetLicense(<[In]> ByRef uuidKey As Guid, <[In], Out> ByRef pulPackageLen As Integer, <[In], Out> ByRef pbPackage As Byte) As Integer

		<PreserveSig> _
		Function ReissueLicense(<[In]> ByRef uuidKey As Guid) As Integer

		<PreserveSig> _
		Function RenewLicense(<[In]> ulInXmrLicenseLen As Integer, <[In]> ByRef pbInXmrLicense As Byte, <[In]> ulEntitlementTokenLen As Integer, <[In]> ByRef pbEntitlementToken As Byte, ByRef pulDescrambleStatus As Integer, <[In], Out> ByRef pulOutXmrLicenseLen As Integer, _
			<[In], Out> ByRef pbOutXmrLicense As Byte) As Integer

		<PreserveSig> _
		Function GetKeyInfo(<[In], Out> ByRef pulKeyInfoLen As Integer, <[In], Out> ByRef pbKeyInfo As Byte) As Integer
	End Interface

	<ComImport, System.Security.SuppressUnmanagedCodeSecurity, Guid("86D979CF-A8A7-4F94-B5FB-14C0ACA68FE6"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)> _
	Public Interface IBDA_WMDRMTuner
		<PreserveSig> _
		Function PurchaseEntitlement(<[In]> ulDialogRequest As Integer, <[In], MarshalAs(UnmanagedType.BStr)> bstrLanguage As String, <[In]> ulPurchaseTokenLen As Integer, <[In]> ByRef pbPurchaseToken As Byte, ByRef pulDescrambleStatus As Integer, <[In], Out> ByRef pulCaptureTokenLen As Integer, _
			<[In], Out> ByRef pbCaptureToken As Byte) As Integer

		<PreserveSig> _
		Function CancelCaptureToken(<[In]> ulCaptureTokenLen As Integer, <[In]> ByRef pbCaptureToken As Byte) As Integer

		<PreserveSig> _
		Function SetPidProtection(<[In]> ulPid As Integer, <[In]> ByRef uuidKey As Guid) As Integer

		<PreserveSig> _
		Function GetPidProtection(<[In]> pulPid As Integer, ByRef uuidKey As Guid) As Integer

		<PreserveSig> _
		Function SetSyncValue(<[In]> ulSyncValue As Integer) As Integer

		<PreserveSig> _
		Function GetStartCodeProfile(<[In], Out> ByRef pulStartCodeProfileLen As Integer, <[In], Out> ByRef pbStartCodeProfile As Byte) As Integer
	End Interface

	<ComImport, System.Security.SuppressUnmanagedCodeSecurity, Guid("13F19604-7D32-4359-93A2-A05205D90AC9"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)> _
	Public Interface IBDA_DigitalDemodulator3
		Inherits IBDA_DigitalDemodulator2

		<PreserveSig> _
		Shadows Function put_ModulationType(<[In]> ByRef pModulationType As ModulationType) As Integer

		<PreserveSig> _
		Shadows Function get_ModulationType(<Out> ByRef pModulationType As ModulationType) As Integer

		<PreserveSig> _
		Shadows Function put_InnerFECMethod(<[In]> ByRef pFECMethod As FECMethod) As Integer

		<PreserveSig> _
		Shadows Function get_InnerFECMethod(<Out> ByRef pFECMethod As FECMethod) As Integer

		<PreserveSig> _
		Shadows Function put_InnerFECRate(<[In]> ByRef pFECRate As BinaryConvolutionCodeRate) As Integer

		<PreserveSig> _
		Shadows Function get_InnerFECRate(<Out> ByRef pFECRate As BinaryConvolutionCodeRate) As Integer

		<PreserveSig> _
		Shadows Function put_OuterFECMethod(<[In]> ByRef pFECMethod As FECMethod) As Integer

		<PreserveSig> _
		Shadows Function get_OuterFECMethod(<Out> ByRef pFECMethod As FECMethod) As Integer

		<PreserveSig> _
		Shadows Function put_OuterFECRate(<[In]> ByRef pFECRate As BinaryConvolutionCodeRate) As Integer

		<PreserveSig> _
		Shadows Function get_OuterFECRate(<Out> ByRef pFECRate As BinaryConvolutionCodeRate) As Integer

		<PreserveSig> _
		Shadows Function put_SymbolRate(<[In]> ByRef pSymbolRate As Integer) As Integer

		<PreserveSig> _
		Shadows Function get_SymbolRate(<Out> ByRef pSymbolRate As Integer) As Integer

		<PreserveSig> _
		Shadows Function put_SpectralInversion(<[In]> ByRef pSpectralInversion As SpectralInversion) As Integer

		<PreserveSig> _
		Shadows Function get_SpectralInversion(<Out> ByRef pSpectralInversion As SpectralInversion) As Integer

		<PreserveSig> _
		Shadows Function put_GuardInterval(<[In]> ByRef pGuardInterval As GuardInterval) As Integer

		<PreserveSig> _
		Shadows Function get_GuardInterval(<[In], Out> ByRef pGuardInterval As GuardInterval) As Integer

		<PreserveSig> _
		Shadows Function put_TransmissionMode(<[In]> ByRef pTransmissionMode As TransmissionMode) As Integer

		<PreserveSig> _
		Shadows Function get_TransmissionMode(<[In], Out> ByRef pTransmissionMode As TransmissionMode) As Integer

		<PreserveSig> _
		Shadows Function put_RollOff(<[In]> ByRef pRollOff As RollOff) As Integer

		<PreserveSig> _
		Shadows Function get_RollOff(<[In], Out> ByRef pRollOff As RollOff) As Integer

		<PreserveSig> _
		Shadows Function put_Pilot(<[In]> ByRef pPilot As Pilot) As Integer

		<PreserveSig> _
		Shadows Function get_Pilot(<[In], Out> ByRef pPilot As Pilot) As Integer

		<PreserveSig> _
		Function put_SignalTimeouts(<[In]> pSignalTimeouts As BDA_SIGNAL_TIMEOUTS) As Integer

		<PreserveSig> _
		Function get_SignalTimeouts(<[In], Out> pSignalTimeouts As BDA_SIGNAL_TIMEOUTS) As Integer

		<PreserveSig> _
		Function put_PLPNumber(<[In]> ByRef pPLPNumber As Integer) As Integer

		<PreserveSig> _
		Function get_PLPNumber(<[In], Out> ByRef pPLPNumber As Integer) As Integer
	End Interface

#End If

    <ComImport(), System.Security.SuppressUnmanagedCodeSecurity(), Guid("1347D106-CF3A-428a-A5CB-AC0D9A2A4338"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)> _
    Public Interface IBDA_SignalStatistics
        <PreserveSig()> _
        Function put_SignalStrength(<[In]()> ByVal lDbStrength As Integer) As Integer

        <PreserveSig()> _
        Function get_SignalStrength(<Out()> ByRef plDbStrength As Integer) As Integer

        <PreserveSig()> _
        Function put_SignalQuality(<[In]()> ByVal lPercentQuality As Integer) As Integer

        <PreserveSig()> _
        Function get_SignalQuality(<Out()> ByRef plPercentQuality As Integer) As Integer

        <PreserveSig()> _
        Function put_SignalPresent(<[In](), MarshalAs(UnmanagedType.U1)> ByVal fPresent As Boolean) As Integer

        <PreserveSig()> _
        Function get_SignalPresent(<Out(), MarshalAs(UnmanagedType.U1)> ByRef pfPresent As Boolean) As Integer

        <PreserveSig()> _
        Function put_SignalLocked(<[In](), MarshalAs(UnmanagedType.U1)> ByVal fLocked As Boolean) As Integer

        <PreserveSig()> _
        Function get_SignalLocked(<Out(), MarshalAs(UnmanagedType.U1)> ByRef pfLocked As Boolean) As Integer

        <PreserveSig()> _
        Function put_SampleTime(<[In]()> ByVal lmsSampleTime As Integer) As Integer

        <PreserveSig()> _
        Function get_SampleTime(<Out()> ByRef plmsSampleTime As Integer) As Integer
    End Interface

    <ComImport(), System.Security.SuppressUnmanagedCodeSecurity(), Guid("79B56888-7FEA-4690-B45D-38FD3C7849BE"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)> _
    Public Interface IBDA_Topology
        <PreserveSig()> _
        Function GetNodeTypes(<Out()> ByRef pulcNodeTypes As Integer, <[In]()> ByVal ulcNodeTypesMax As Integer, <Out(), MarshalAs(UnmanagedType.LPArray, ArraySubType:=UnmanagedType.I4, SizeParamIndex:=1)> ByVal rgulNodeTypes As Integer()) As Integer

        <PreserveSig()> _
        Function GetNodeDescriptors(<Out()> ByRef ulcNodeDescriptors As Integer, <[In]()> ByVal ulcNodeDescriptorsMax As Integer, <Out(), MarshalAs(UnmanagedType.LPArray, ArraySubType:=UnmanagedType.LPStruct, SizeParamIndex:=1)> ByVal rgNodeDescriptors As BDANodeDescriptor()) As Integer

        <PreserveSig()> _
        Function GetNodeInterfaces(<[In]()> ByVal ulNodeType As Integer, <Out()> ByRef pulcInterfaces As Integer, <[In]()> ByVal ulcInterfacesMax As Integer, <Out(), MarshalAs(UnmanagedType.LPArray, ArraySubType:=UnmanagedType.LPStruct, SizeParamIndex:=2)> ByVal rgguidInterfaces As Guid()) As Integer

        <PreserveSig()> _
        Function GetPinTypes(<Out()> ByRef pulcPinTypes As Integer, <[In]()> ByVal ulcPinTypesMax As Integer, <Out(), MarshalAs(UnmanagedType.LPArray, ArraySubType:=UnmanagedType.I4, SizeParamIndex:=1)> ByVal rgulPinTypes As Integer()) As Integer

        <PreserveSig()> _
        Function GetTemplateConnections(<Out()> ByRef pulcConnections As Integer, <[In]()> ByVal ulcConnectionsMax As Integer, <Out(), MarshalAs(UnmanagedType.LPArray, ArraySubType:=UnmanagedType.LPStruct, SizeParamIndex:=1)> ByVal rgConnections As BDATemplateConnection()) As Integer

        <PreserveSig()> _
        Function CreatePin(<[In]()> ByVal ulPinType As Integer, <Out()> ByRef pulPinId As Integer) As Integer

        <PreserveSig()> _
        Function DeletePin(<[In]()> ByVal ulPinId As Integer) As Integer

        <PreserveSig()> _
        Function SetMediaType(<[In]()> ByVal ulPinId As Integer, <[In](), MarshalAs(UnmanagedType.LPStruct)> ByVal pMediaType As AMMediaType) As Integer

        <PreserveSig()> _
        Function SetMedium(<[In]()> ByVal ulPinId As Integer, <[In]()> ByVal pMedium As RegPinMedium) As Integer

        <PreserveSig()> _
        Function CreateTopology(<[In]()> ByVal ulInputPinId As Integer, <[In]()> ByVal ulOutputPinId As Integer) As Integer

        ' IUnknown
        <PreserveSig()> _
        Function GetControlNode(<[In]()> ByVal ulInputPinId As Integer, <[In]()> ByVal ulOutputPinId As Integer, <[In]()> ByVal ulNodeType As Integer, <Out(), MarshalAs(UnmanagedType.IUnknown)> ByRef ppControlNode As Object) As Integer

    End Interface

    <ComImport(), System.Security.SuppressUnmanagedCodeSecurity(), Guid("06FB45C1-693C-4ea7-B79F-7A6A54D8DEF2"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)> _
    Public Interface IFrequencyMap
        <PreserveSig()> _
        Function get_FrequencyMapping(<Out()> ByRef ulCount As Integer, <Out()> ByRef ppulList As IntPtr) As Integer

        <PreserveSig()> _
        Function put_FrequencyMapping(<[In]()> ByVal ulCount As Integer, <[In](), MarshalAs(UnmanagedType.LPArray, ArraySubType:=UnmanagedType.I4)> ByVal pList As Integer()) As Integer

        <PreserveSig()> _
        Function get_CountryCode(<Out()> ByRef pulCountryCode As Integer) As Integer

        <PreserveSig()> _
        Function put_CountryCode(<[In]()> ByVal ulCountryCode As Integer) As Integer

        <PreserveSig()> _
        Function get_DefaultFrequencyMapping(<[In]()> ByVal ulCountryCode As Integer, <Out()> ByRef pulCount As Integer, <Out()> ByRef ppulList As IntPtr) As Integer

        <PreserveSig()> _
        Function get_CountryCodeList(<Out()> ByRef pulCount As Integer, <Out()> ByRef ppulList As IntPtr) As Integer

    End Interface

    <ComImport(), System.Security.SuppressUnmanagedCodeSecurity(), Guid("DDF15B12-BD25-11d2-9CA0-00C04F7971E0"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)> _
    Public Interface IBDA_AutoDemodulate
        <PreserveSig()> _
        Function put_AutoDemodulate() As Integer
    End Interface

    <ComImport(), System.Security.SuppressUnmanagedCodeSecurity(), Guid("FD0A5AF3-B41D-11d2-9C95-00C04F7971E0"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)> _
    Public Interface IBDA_DeviceControl
        <PreserveSig()> _
        Function StartChanges() As Integer

        <PreserveSig()> _
        Function CheckChanges() As Integer

        <PreserveSig()> _
        Function CommitChanges() As Integer

        <PreserveSig()> _
        Function GetChangeState(<Out()> ByRef pState As BDAChangeState) As Integer
    End Interface

    <ComImport(), System.Security.SuppressUnmanagedCodeSecurity(), Guid("EF30F379-985B-4d10-B640-A79D5E04E1E0"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)> _
    Public Interface IBDA_DigitalDemodulator
        <PreserveSig()> _
        Function put_ModulationType(<[In]()> ByRef pModulationType As ModulationType) As Integer

        <PreserveSig()> _
        Function get_ModulationType(<Out()> ByRef pModulationType As ModulationType) As Integer

        <PreserveSig()> _
        Function put_InnerFECMethod(<[In]()> ByRef pFECMethod As FECMethod) As Integer

        <PreserveSig()> _
        Function get_InnerFECMethod(<Out()> ByRef pFECMethod As FECMethod) As Integer

        <PreserveSig()> _
        Function put_InnerFECRate(<[In]()> ByRef pFECRate As BinaryConvolutionCodeRate) As Integer

        <PreserveSig()> _
        Function get_InnerFECRate(<Out()> ByRef pFECRate As BinaryConvolutionCodeRate) As Integer

        <PreserveSig()> _
        Function put_OuterFECMethod(<[In]()> ByRef pFECMethod As FECMethod) As Integer

        <PreserveSig()> _
        Function get_OuterFECMethod(<Out()> ByRef pFECMethod As FECMethod) As Integer

        <PreserveSig()> _
        Function put_OuterFECRate(<[In]()> ByRef pFECRate As BinaryConvolutionCodeRate) As Integer

        <PreserveSig()> _
        Function get_OuterFECRate(<Out()> ByRef pFECRate As BinaryConvolutionCodeRate) As Integer

        <PreserveSig()> _
        Function put_SymbolRate(<[In]()> ByRef pSymbolRate As Integer) As Integer

        <PreserveSig()> _
        Function get_SymbolRate(<Out()> ByRef pSymbolRate As Integer) As Integer

        <PreserveSig()> _
        Function put_SpectralInversion(<[In]()> ByRef pSpectralInversion As SpectralInversion) As Integer

        <PreserveSig()> _
        Function get_SpectralInversion(<Out()> ByRef pSpectralInversion As SpectralInversion) As Integer
    End Interface

    <ComImport(), System.Security.SuppressUnmanagedCodeSecurity(), Guid("71985F43-1CA1-11d3-9CC8-00C04F7971E0"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)> _
    Public Interface IBDA_EthernetFilter
        <PreserveSig()> _
        Function GetMulticastListSize(ByRef pulcbAddresses As Integer) As Integer

        <PreserveSig()> _
        Function PutMulticastList(ByVal ulcbAddresses As Integer, ByVal pAddressList As IntPtr) As Integer

        <PreserveSig()> _
        Function GetMulticastList(ByRef pulcbAddresses As Integer, ByVal pAddressList As IntPtr) As Integer

        <PreserveSig()> _
        Function PutMulticastMode(ByVal ulModeMask As MulticastMode) As Integer

        <PreserveSig()> _
        Function GetMulticastMode(ByRef pulModeMask As MulticastMode) As Integer

    End Interface

    <ComImport(), System.Security.SuppressUnmanagedCodeSecurity(), Guid("71985F47-1CA1-11d3-9CC8-00C04F7971E0"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)> _
    Public Interface IBDA_FrequencyFilter
        <PreserveSig()> _
        Function put_Autotune(<[In]()> ByVal ulTransponder As Integer) As Integer

        <PreserveSig()> _
        Function get_Autotune(<Out()> ByRef pulTransponder As Integer) As Integer

        <PreserveSig()> _
        Function put_Frequency(<[In]()> ByVal ulFrequency As Integer) As Integer

        <PreserveSig()> _
        Function get_Frequency(<Out()> ByRef pulFrequency As Integer) As Integer

        <PreserveSig()> _
        Function put_Polarity(<[In]()> ByVal Polarity As Polarisation) As Integer

        <PreserveSig()> _
        Function get_Polarity(<Out()> ByRef pPolarity As Polarisation) As Integer

        <PreserveSig()> _
        Function put_Range(<[In]()> ByVal ulRange As Integer) As Integer

        <PreserveSig()> _
        Function get_Range(<Out()> ByRef pulRange As Integer) As Integer

        <PreserveSig()> _
        Function put_Bandwidth(<[In]()> ByVal ulBandwidth As Integer) As Integer

        <PreserveSig()> _
        Function get_Bandwidth(<Out()> ByRef pulBandwidth As Integer) As Integer

        <PreserveSig()> _
        Function put_FrequencyMultiplier(<[In]()> ByVal ulMultiplier As Integer) As Integer

        <PreserveSig()> _
        Function get_FrequencyMultiplier(<Out()> ByRef pulMultiplier As Integer) As Integer
    End Interface

    <ComImport(), System.Security.SuppressUnmanagedCodeSecurity(), Guid("3F4DC8E2-4050-11d3-8F4B-00C04F7971E2"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown), Obsolete("IBDA_IPSinkControl is no longer being supported for Ring 3 clients. Use the BDA_IPSinkInfo interface instead.")> _
    Public Interface IBDA_IPSinkControl
        ' BYTE **
        <PreserveSig()> _
        Function GetMulticastList(ByRef pulcbSize As Integer, ByRef pbBuffer As IntPtr) As Integer

        ' BYTE **
        <PreserveSig()> _
        Function GetAdapterIPAddress(ByRef pulcbSize As Integer, ByRef pbBuffer As IntPtr) As Integer

    End Interface

    <ComImport(), System.Security.SuppressUnmanagedCodeSecurity(), Guid("A750108F-492E-4d51-95F7-649B23FF7AD7"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)> _
    Public Interface IBDA_IPSinkInfo

        ' BYTE **
        <PreserveSig()> _
        Function get_MulticastList(ByRef pulcbAddresses As Integer, ByRef ppbAddressList As IntPtr) As Integer

        <PreserveSig()> _
        Function get_AdapterIPAddress(<MarshalAs(UnmanagedType.BStr)> ByRef pbstrBuffer As String) As Integer

        <PreserveSig()> _
        Function get_AdapterDescription(<MarshalAs(UnmanagedType.BStr)> ByRef pbstrBuffer As String) As Integer
    End Interface

    <ComImport(), System.Security.SuppressUnmanagedCodeSecurity(), Guid("71985F44-1CA1-11d3-9CC8-00C04F7971E0"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)> _
    Public Interface IBDA_IPV4Filter
        <PreserveSig()> _
        Function GetMulticastListSize(ByRef pulcbAddresses As Integer) As Integer

        <PreserveSig()> _
        Function PutMulticastList(ByVal ulcbAddresses As Integer, ByVal pAddressList As IntPtr) As Integer

        <PreserveSig()> _
        Function GetMulticastList(ByRef pulcbAddresses As Integer, ByVal pAddressList As IntPtr) As Integer

        <PreserveSig()> _
        Function PutMulticastMode(ByVal ulModeMask As MulticastMode) As Integer

        <PreserveSig()> _
        Function GetMulticastMode(ByRef pulModeMask As MulticastMode) As Integer

    End Interface

    <ComImport(), System.Security.SuppressUnmanagedCodeSecurity(), Guid("E1785A74-2A23-4fb3-9245-A8F88017EF33"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)> _
    Public Interface IBDA_IPV6Filter
        <PreserveSig()> _
        Function GetMulticastListSize(ByRef pulcbAddresses As Integer) As Integer

        ' BYTE []
        <PreserveSig()> _
        Function PutMulticastList(ByVal ulcbAddresses As Integer, ByVal pAddressList As IntPtr) As Integer

        ' BYTE []
        <PreserveSig()> _
        Function GetMulticastList(ByRef pulcbAddresses As Integer, ByVal pAddressList As IntPtr) As Integer

        <PreserveSig()> _
        Function PutMulticastMode(ByVal ulModeMask As MulticastMode) As Integer

        <PreserveSig()> _
        Function GetMulticastMode(ByRef pulModeMask As MulticastMode) As Integer
    End Interface

    <ComImport(), System.Security.SuppressUnmanagedCodeSecurity(), Guid("992CF102-49F9-4719-A664-C4F23E2408F4"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)> _
    Public Interface IBDA_LNBInfo
        <PreserveSig()> _
        Function put_LocalOscilatorFrequencyLowBand(<[In]()> ByVal ulLOFLow As Integer) As Integer

        <PreserveSig()> _
        Function get_LocalOscilatorFrequencyLowBand(<Out()> ByRef pulLOFLow As Integer) As Integer

        <PreserveSig()> _
        Function put_LocalOscilatorFrequencyHighBand(<[In]()> ByVal ulLOFHigh As Integer) As Integer

        <PreserveSig()> _
        Function get_LocalOscilatorFrequencyHighBand(<Out()> ByRef pulLOFHigh As Integer) As Integer

        <PreserveSig()> _
        Function put_HighLowSwitchFrequency(<[In]()> ByVal ulSwitchFrequency As Integer) As Integer

        <PreserveSig()> _
        Function get_HighLowSwitchFrequency(<Out()> ByRef pulSwitchFrequency As Integer) As Integer
    End Interface

    <ComImport(), System.Security.SuppressUnmanagedCodeSecurity(), Guid("afb6c2a1-2c41-11d3-8a60-0000f81e0e4a"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)> _
    Public Interface IMPEG2PIDMap
        <PreserveSig()> _
        Function MapPID(<[In]()> ByVal culPID As Integer, <[In](), MarshalAs(UnmanagedType.LPArray)> ByVal pulPID As Integer(), <[In]()> ByVal MediaSampleContent As MediaSampleContent) As Integer

        <PreserveSig()> _
        Function UnmapPID(<[In]()> ByVal culPID As Integer, <[In](), MarshalAs(UnmanagedType.LPArray)> ByVal pulPID As Integer()) As Integer

#If ALLOW_UNTESTED_INTERFACES Then
		<PreserveSig, Obsolete("Because of bug in DS 9.0c, you can't get the PID map from .NET", False)> _
		Function EnumPIDMap(<Out> ByRef pIEnumPIDMap As IEnumPIDMap) As Integer
#Else
        Function EnumPIDMap(<Out()> ByRef pIEnumPIDMap As Object) As Integer
#End If

    End Interface

    <ComImport(), System.Security.SuppressUnmanagedCodeSecurity(), Guid("D2F1644B-B409-11d2-BC69-00A0C9EE9E16"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)> _
    Public Interface IBDA_SignalProperties
        <PreserveSig()> _
        Function PutNetworkType(<[In](), MarshalAs(UnmanagedType.LPStruct)> ByVal guidNetworkType As Guid) As Integer

        <PreserveSig()> _
        Function GetNetworkType(<Out()> ByRef pguidNetworkType As Guid) As Integer

        <PreserveSig()> _
        Function PutSignalSource(<[In]()> ByVal ulSignalSource As Integer) As Integer

        <PreserveSig()> _
        Function GetSignalSource(<Out()> ByRef pulSignalSource As Integer) As Integer

        <PreserveSig()> _
        Function PutTuningSpace(<[In](), MarshalAs(UnmanagedType.LPStruct)> ByVal guidTuningSpace As Guid) As Integer

        <PreserveSig()> _
        Function GetTuningSpace(<Out()> ByRef pguidTuingSpace As Guid) As Integer
    End Interface

    <ComImport(), System.Security.SuppressUnmanagedCodeSecurity(), Guid("4B2BD7EA-8347-467b-8DBF-62F784929CC3"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)> _
    Public Interface ICCSubStreamFiltering
        <PreserveSig()> _
        Function get_SubstreamTypes(<Out()> ByRef Types As CCSubstreamService) As Integer

        <PreserveSig()> _
        Function put_SubstreamTypes(<[In]()> ByVal Types As CCSubstreamService) As Integer
    End Interface

End Namespace
