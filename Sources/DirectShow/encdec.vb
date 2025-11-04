
Imports System.Runtime.InteropServices

Namespace BDA

    <ComImport(), Guid("C4C4C481-0049-4E2B-98FB-9537F6CE516D")> _
    Public Class ETFilterEncProperties
    End Class

    <ComImport(), Guid("C4C4C491-0049-4E2B-98FB-9537F6CE516D")> _
    Public Class ETFilterTagProperties
    End Class

    <ComImport(), Guid("C4C4C482-0049-4E2B-98FB-9537F6CE516D")> _
    Public Class DTFilterEncProperties
    End Class

    <ComImport(), Guid("C4C4C492-0049-4E2B-98FB-9537F6CE516D")> _
    Public Class DTFilterTagProperties
    End Class

    <ComImport(), Guid("C4C4C483-0049-4E2B-98FB-9537F6CE516D")> _
    Public Class XDSCodecProperties
    End Class

    <ComImport(), Guid("C4C4C493-0049-4E2B-98FB-9537F6CE516D")> _
    Public Class XDSCodecTagProperties
    End Class

    <ComImport(), Guid("C4C4C4F4-0049-4E2B-98FB-9537F6CE516D")> _
    Public Class CXDSData
    End Class

    <ComImport(), Guid("C4C4C4F3-0049-4E2B-98FB-9537F6CE516D")> _
    Public Class XDSCodec
    End Class


#If ALLOW_UNTESTED_INTERFACES Then

	Public Enum FormatNotSupportedEvents
		Clear = 0
		NotSupported = 1
	End Enum

	<StructLayout(LayoutKind.Sequential)> _
	Public Structure WMDRMProtectionInfo
		<MarshalAs(UnmanagedType.LPWStr, SizeConst := 25)> _
		Private wszKID As String
		Public qwCounter As Long
		Public qwIndex As Long
		Public bOffset As Byte
	End Structure

	Public Class BadSampleInfo
		Public hrReason As Integer
	End Class

	Public Enum COPPEventBlockReason
		Unknown = -1
		BadDriver = 0
		NoCardHDCPSupport = 1
		NoMonitorHDCPSupport = 2
		BadCertificate = 3
		InvalidBusProtection = 4
		AeroGlassOff = 5
		RogueApp = 6
		ForbiddenVideo = 7
		Activate = 8
		DigitalAudioUnprotected = 9
	End Enum

	Public Enum LicenseEventBlockReason
		BadLicense = 0
		NeedIndiv = 1
		Expired = 2
		NeedActivation = 3
		ExtenderBlocked = 4
	End Enum

	Public Enum CPEventBitShift
		Ratings = 0
		COPP
		License
		Rollback
		SAC
		DownRes
		StubLib
		UntrustedGraph
		PendingCertificate
		NoPlayReady
	End Enum

	Public Enum CPEvents
		None = 0
		Ratings
		COPP
		License
		Rollback
		SAC
		DownRes
		StubLib
		UntrustedGraph
		ProtectWindowed
	End Enum

	Public Enum EncDecEvents
		CPEvent = 0
		RecordingStatus
	End Enum

	Public Enum CPRecordingStatus
		Stopped = 0
		Started = 1
	End Enum

	Public Enum RevokedComponent
		COPP = 0
		SAC
		APPStub
		SecurePipeline
		MaxTypes
	End Enum

	Public Enum EnTag_Mode
		Remove = &H0
		Once = &H1
		Repeat = &H2
	End Enum

	Public Enum DownResEventParam
		Always = 0
		InWindowOnly = 1
		Undefined = 2
	End Enum

#End If


    Public Enum ProtType
        None = 0
        Free = 1
        Once = 2
        Never = 3
        NeverReally = 4
        NoMore = 5
        FreeCit = 6
        BF = 7
        CnRecordingStop = 8
        FreeSecure = 9
        Invalid = 50
    End Enum

    Public NotInheritable Class EventID
        Private Sub New()
        End Sub

        Public Shared ReadOnly XDSCodecNewXDSRating As New Guid(&HC4C4C4E0UI, &H49, &H4E2B, &H98, &HFB, &H95, _
         &H37, &HF6, &HCE, &H51, &H6D)

        Public Shared ReadOnly XDSCodecDuplicateXDSRating As New Guid(&HC4C4C4DFUI, &H49, &H4E2B, &H98, &HFB, &H95, _
         &H37, &HF6, &HCE, &H51, &H6D)

        Public Shared ReadOnly XDSCodecNewXDSPacket As New Guid(&HC4C4C4E1UI, &H49, &H4E2B, &H98, &HFB, &H95, _
         &H37, &HF6, &HCE, &H51, &H6D)

        Public Shared ReadOnly DTFilterRatingChange As New Guid(&HC4C4C4E2UI, &H49, &H4E2B, &H98, &HFB, &H95, _
         &H37, &HF6, &HCE, &H51, &H6D)

        Public Shared ReadOnly DTFilterRatingsBlock As New Guid(&HC4C4C4E3UI, &H49, &H4E2B, &H98, &HFB, &H95, _
         &H37, &HF6, &HCE, &H51, &H6D)

        Public Shared ReadOnly DTFilterRatingsUnblock As New Guid(&HC4C4C4E4UI, &H49, &H4E2B, &H98, &HFB, &H95, _
         &H37, &HF6, &HCE, &H51, &H6D)

        Public Shared ReadOnly DTFilterXDSPacket As New Guid(&HC4C4C4E5UI, &H49, &H4E2B, &H98, &HFB, &H95, _
         &H37, &HF6, &HCE, &H51, &H6D)

        Public Shared ReadOnly ETFilterEncryptionOn As New Guid(&HC4C4C4E6UI, &H49, &H4E2B, &H98, &HFB, &H95, _
         &H37, &HF6, &HCE, &H51, &H6D)

        Public Shared ReadOnly ETFilterEncryptionOff As New Guid(&HC4C4C4E7UI, &H49, &H4E2B, &H98, &HFB, &H95, _
         &H37, &HF6, &HCE, &H51, &H6D)

        Public Shared ReadOnly DTFilterCOPPUnblock As New Guid(&HC4C4C4E8UI, &H49, &H4E2B, &H98, &HFB, &H95, _
         &H37, &HF6, &HCE, &H51, &H6D)

        Public Shared ReadOnly EncDecFilterError As New Guid(&HC4C4C4E9UI, &H49, &H4E2B, &H98, &HFB, &H95, _
         &H37, &HF6, &HCE, &H51, &H6D)

        Public Shared ReadOnly DTFilterCOPPBlock As New Guid(&HC4C4C4EAUI, &H49, &H4E2B, &H98, &HFB, &H95, _
         &H37, &HF6, &HCE, &H51, &H6D)

        Public Shared ReadOnly ETFilterCopyOnce As New Guid(&HC4C4C4EBUI, &H49, &H4E2B, &H98, &HFB, &H95, _
         &H37, &HF6, &HCE, &H51, &H6D)

        Public Shared ReadOnly ETFilterCopyNever As New Guid(&HC4C4C4F0UI, &H49, &H4E2B, &H98, &HFB, &H95, _
         &H37, &HF6, &HCE, &H51, &H6D)

        Public Shared ReadOnly DTFilterDataFormatOK As New Guid(&HC4C4C4ECUI, &H49, &H4E2B, &H98, &HFB, &H95, _
         &H37, &HF6, &HCE, &H51, &H6D)

        Public Shared ReadOnly DTFilterDataFormatFailure As New Guid(&HC4C4C4EDUI, &H49, &H4E2B, &H98, &HFB, &H95, _
         &H37, &HF6, &HCE, &H51, &H6D)

        Public Shared ReadOnly ETDTFilterLicenseOK As New Guid(&HC4C4C4EEUI, &H49, &H4E2B, &H98, &HFB, &H95, _
         &H37, &HF6, &HCE, &H51, &H6D)

        Public Shared ReadOnly ETDTFilterLicenseFailure As New Guid(&HC4C4C4EFUI, &H49, &H4E2B, &H98, &HFB, &H95, _
         &H37, &HF6, &HCE, &H51, &H6D)

        Public Shared ReadOnly EncDecFilterEvent As New Guid(&H4A1B465B, &HFB9, &H4159, &HAF, &HBD, &HE3, _
         &H30, &H6, &HA0, &HF9, &HF4)

        Public Shared ReadOnly FormatNotSupportedEvent As New Guid(&H24B2280A, &HB2AA, &H4777, &HBF, &H65, &H63, _
         &HF3, &H5E, &H7B, &H2, &H4A)

        Public Shared ReadOnly DemultiplexerFilterDiscontinuity As New Guid(&H16155770, &HAED5, &H475C, &HBB, &H98, &H95, _
         &HA3, &H30, &H70, &HDF, &HC)
    End Class


#If ALLOW_UNTESTED_INTERFACES Then

	<ComImport, System.Security.SuppressUnmanagedCodeSecurity, Guid("FAF37694-909C-49cd-886F-C7382E5DB596"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)> _
	Public Interface IDTFilterBlockedOverlay
		<PreserveSig> _
		Function SetOverlay(dwOverlayCause As Integer) As Integer

		<PreserveSig> _
		Function ClearOverlay(dwOverlayCause As Integer) As Integer

		<PreserveSig> _
		Function GetOverlay(ByRef pdwOverlayCause As Integer) As Integer

	End Interface

	<ComImport, System.Security.SuppressUnmanagedCodeSecurity, Guid("C4C4C4C2-0049-4E2B-98FB-9537F6CE516D"), InterfaceType(ComInterfaceType.InterfaceIsDual)> _
	Public Interface IDTFilterEvents
	End Interface

	<ComImport, System.Security.SuppressUnmanagedCodeSecurity, Guid("C4C4C4C1-0049-4E2B-98FB-9537F6CE516D"), InterfaceType(ComInterfaceType.InterfaceIsDual)> _
	Public Interface IETFilterEvents
	End Interface

	<ComImport, System.Security.SuppressUnmanagedCodeSecurity, Guid("C4C4C4C3-0049-4E2B-98FB-9537F6CE516D"), InterfaceType(ComInterfaceType.InterfaceIsDual)> _
	Public Interface IXDSCodecEvents
	End Interface

	<ComImport, System.Security.SuppressUnmanagedCodeSecurity, Guid("C4C4C4B1-0049-4E2B-98FB-9537F6CE516D"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)> _
	Public Interface IETFilter
		<PreserveSig> _
		Function get_EvalRatObjOK(ByRef pHrCoCreateRetVal As Integer) As Integer

		<PreserveSig> _
		Function GetCurrRating(ByRef pEnSystem As EnTvRat_System, ByRef pEnRating As EnTvRat_GenericLevel, ByRef plbfEnAttr As BfEnTvRat_GenericAttributes) As Integer

		<PreserveSig> _
		Function GetCurrLicenseExpDate(protType As ProtType, ByRef lpDateTime As Integer) As Integer

		<PreserveSig> _
		Function GetLastErrorCode() As Integer

		<PreserveSig> _
		Function SetRecordingOn(<MarshalAs(UnmanagedType.Bool)> fRecState As Boolean) As Integer
	End Interface

	<ComImport, System.Security.SuppressUnmanagedCodeSecurity, Guid("C4C4C4B3-0049-4E2B-98FB-9537F6CE516D"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)> _
	Public Interface IXDSCodec
		<PreserveSig> _
		Function get_XDSToRatObjOK(ByRef pHrCoCreateRetVal As Integer) As Integer

		<PreserveSig> _
		Function put_CCSubstreamService(SubstreamMask As Integer) As Integer

		<PreserveSig> _
		Function get_CCSubstreamService(ByRef pSubstreamMask As Integer) As Integer

		<PreserveSig> _
		Function GetContentAdvisoryRating(ByRef pRat As Integer, ByRef pPktSeqID As Integer, ByRef pCallSeqID As Integer, ByRef pTimeStart As Long, ByRef pTimeEnd As Long) As Integer

		<PreserveSig> _
		Function GetXDSPacket(ByRef pXDSClassPkt As Integer, ByRef pXDSTypePkt As Integer, <MarshalAs(UnmanagedType.BStr)> ByRef pBstrXDSPkt As String, ByRef pPktSeqID As Integer, ByRef pCallSeqID As Integer, ByRef pTimeStart As Long, _
			ByRef pTimeEnd As Long) As Integer

		<PreserveSig> _
		Function GetCurrLicenseExpDate(protType As ProtType, ByRef lpDateTime As Integer) As Integer

		<PreserveSig> _
		Function GetLastErrorCode() As Integer
	End Interface

#End If

    <ComImport(), System.Security.SuppressUnmanagedCodeSecurity(), Guid("C4C4C4D3-0049-4E2B-98FB-9537F6CE516D"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)> _
    Public Interface IXDSCodecConfig
        <PreserveSig()> _
        Function GetSecureChannelObject(<MarshalAs(UnmanagedType.IUnknown)> ByRef ppUnkDRMSecureChannel As Object) As Integer

        <PreserveSig()> _
        Function SetPauseBufferTime(ByVal dwPauseBufferTime As Integer) As Integer
    End Interface

    <ComImport(), System.Security.SuppressUnmanagedCodeSecurity(), Guid("C4C4C4B2-0049-4E2B-98FB-9537F6CE516D"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)> _
    Public Interface IDTFilter
        <PreserveSig()> _
        Function get_EvalRatObjOK(ByRef pHrCoCreateRetVal As Integer) As Integer

        <PreserveSig()> _
        Function GetCurrRating(ByRef pEnSystem As EnTvRat_System, ByRef pEnRating As EnTvRat_GenericLevel, ByRef plbfEnAttr As BfEnTvRat_GenericAttributes) As Integer

        <PreserveSig()> _
        Function get_BlockedRatingAttributes(ByVal enSystem As EnTvRat_System, ByVal enLevel As EnTvRat_GenericLevel, ByRef plbfEnAttr As BfEnTvRat_GenericAttributes) As Integer

        <PreserveSig()> _
        Function put_BlockedRatingAttributes(ByVal enSystem As EnTvRat_System, ByVal enLevel As EnTvRat_GenericLevel, ByVal lbfAttrs As BfEnTvRat_GenericAttributes) As Integer

        <PreserveSig()> _
        Function get_BlockUnRated(<MarshalAs(UnmanagedType.Bool)> ByRef pfBlockUnRatedShows As Boolean) As Integer

        <PreserveSig()> _
        Function put_BlockUnRated(<MarshalAs(UnmanagedType.Bool)> ByVal fBlockUnRatedShows As Boolean) As Integer

        <PreserveSig()> _
        Function get_BlockUnRatedDelay(ByRef pmsecsDelayBeforeBlock As Integer) As Integer

        <PreserveSig()> _
        Function put_BlockUnRatedDelay(ByVal msecsDelayBeforeBlock As Integer) As Integer
    End Interface

    <ComImport(), System.Security.SuppressUnmanagedCodeSecurity(), Guid("C4C4C4B4-0049-4E2B-98FB-9537F6CE516D"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)> _
    Public Interface IDTFilter2
        Inherits IDTFilter

        <PreserveSig()> _
        Shadows Function get_EvalRatObjOK(ByRef pHrCoCreateRetVal As Integer) As Integer

        <PreserveSig()> _
        Shadows Function GetCurrRating(ByRef pEnSystem As EnTvRat_System, ByRef pEnRating As EnTvRat_GenericLevel, ByRef plbfEnAttr As BfEnTvRat_GenericAttributes) As Integer

        <PreserveSig()> _
        Shadows Function get_BlockedRatingAttributes(ByVal enSystem As EnTvRat_System, ByVal enLevel As EnTvRat_GenericLevel, ByRef plbfEnAttr As BfEnTvRat_GenericAttributes) As Integer

        <PreserveSig()> _
        Shadows Function put_BlockedRatingAttributes(ByVal enSystem As EnTvRat_System, ByVal enLevel As EnTvRat_GenericLevel, ByVal lbfAttrs As BfEnTvRat_GenericAttributes) As Integer

        <PreserveSig()> _
        Shadows Function get_BlockUnRated(<MarshalAs(UnmanagedType.Bool)> ByRef pfBlockUnRatedShows As Boolean) As Integer

        <PreserveSig()> _
        Shadows Function put_BlockUnRated(<MarshalAs(UnmanagedType.Bool)> ByVal fBlockUnRatedShows As Boolean) As Integer

        <PreserveSig()> _
        Shadows Function get_BlockUnRatedDelay(ByRef pmsecsDelayBeforeBlock As Integer) As Integer

        <PreserveSig()> _
        Shadows Function put_BlockUnRatedDelay(ByVal msecsDelayBeforeBlock As Integer) As Integer

        <PreserveSig()> _
        Function get_ChallengeUrl(<MarshalAs(UnmanagedType.BStr)> ByRef pbstrChallengeUrl As String) As Integer

        <PreserveSig()> _
        Function GetCurrLicenseExpDate(ByVal protType As ProtType, ByRef lpDateTime As Integer) As Integer

        <PreserveSig()> _
        Function GetLastErrorCode() As Integer

    End Interface

    <ComImport(), System.Security.SuppressUnmanagedCodeSecurity(), Guid("C4C4C4D1-0049-4E2B-98FB-9537F6CE516D"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)> _
    Public Interface IETFilterConfig
        <PreserveSig()> _
        Function InitLicense(ByVal LicenseId As Integer) As Integer

        <PreserveSig()> _
        Function GetSecureChannelObject(<MarshalAs(UnmanagedType.IUnknown)> ByRef ppUnkDRMSecureChannel As Object) As Integer
    End Interface

    <ComImport(), System.Security.SuppressUnmanagedCodeSecurity(), Guid("C4C4C4D2-0049-4E2B-98FB-9537F6CE516D"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)> _
    Public Interface IDTFilterConfig
        <PreserveSig()> _
        Function GetSecureChannelObject(<MarshalAs(UnmanagedType.IUnknown)> ByRef ppUnkDRMSecureChannel As Object) As Integer
    End Interface

    <ComImport(), System.Security.SuppressUnmanagedCodeSecurity(), Guid("513998cc-e929-4cdf-9fbd-bad1e0314866"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)> _
    Public Interface IDTFilter3
        Inherits IDTFilter2

        <PreserveSig()> _
        Shadows Function get_EvalRatObjOK(ByRef pHrCoCreateRetVal As Integer) As Integer

        <PreserveSig()> _
        Shadows Function GetCurrRating(ByRef pEnSystem As EnTvRat_System, ByRef pEnRating As EnTvRat_GenericLevel, ByRef plbfEnAttr As BfEnTvRat_GenericAttributes) As Integer

        <PreserveSig()> _
        Shadows Function get_BlockedRatingAttributes(ByVal enSystem As EnTvRat_System, ByVal enLevel As EnTvRat_GenericLevel, ByRef plbfEnAttr As BfEnTvRat_GenericAttributes) As Integer

        <PreserveSig()> _
        Shadows Function put_BlockedRatingAttributes(ByVal enSystem As EnTvRat_System, ByVal enLevel As EnTvRat_GenericLevel, ByVal lbfAttrs As BfEnTvRat_GenericAttributes) As Integer

        <PreserveSig()> _
        Shadows Function get_BlockUnRated(<MarshalAs(UnmanagedType.Bool)> ByRef pfBlockUnRatedShows As Boolean) As Integer

        <PreserveSig()> _
        Shadows Function put_BlockUnRated(<MarshalAs(UnmanagedType.Bool)> ByVal fBlockUnRatedShows As Boolean) As Integer

        <PreserveSig()> _
        Shadows Function get_BlockUnRatedDelay(ByRef pmsecsDelayBeforeBlock As Integer) As Integer

        <PreserveSig()> _
        Shadows Function put_BlockUnRatedDelay(ByVal msecsDelayBeforeBlock As Integer) As Integer

        <PreserveSig()> _
        Shadows Function get_ChallengeUrl(<MarshalAs(UnmanagedType.BStr)> ByRef pbstrChallengeUrl As String) As Integer

        <PreserveSig()> _
        Shadows Function GetCurrLicenseExpDate(ByVal protType As ProtType, ByRef lpDateTime As Integer) As Integer

        <PreserveSig()> _
        Shadows Function GetLastErrorCode() As Integer

        <PreserveSig()> _
        Function GetProtectionType(ByRef pProtectionType As ProtType) As Integer

        <PreserveSig()> _
        Function LicenseHasExpirationDate(<MarshalAs(UnmanagedType.Bool)> ByRef pfLicenseHasExpirationDate As Boolean) As Integer

        <PreserveSig()> _
        Function SetRights(<MarshalAs(UnmanagedType.BStr)> ByVal bstrRights As String) As Integer
    End Interface


End Namespace
