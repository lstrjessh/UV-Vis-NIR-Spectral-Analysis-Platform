
Imports System.Runtime.InteropServices

<Flags> _
Public Enum AMExtendedSeekingCapabilities
	None = 0
	CanSeek = 1
	CanScan = 2
	MarkerSeek = 4
	ScanWithoutClock = 8
	NoStandardRepaint = 16
	Buffering = 32
	SendsVideoFrameReady = 64
End Enum


#If ALLOW_UNTESTED_INTERFACES Then

<ComImport, System.Security.SuppressUnmanagedCodeSecurity, Guid("FA2AA8F1-8B62-11D0-A520-000000000000"), InterfaceType(ComInterfaceType.InterfaceIsDual)> _
Public Interface IAMNetShowConfig
	<PreserveSig> _
	Function get_BufferingTime(ByRef pBufferingTime As Double) As Integer

	<PreserveSig> _
	Function put_BufferingTime(BufferingTime As Double) As Integer

	<PreserveSig> _
	Function get_UseFixedUDPPort(<MarshalAs(UnmanagedType.VariantBool)> ByRef pUseFixedUDPPort As Boolean) As Integer

	<PreserveSig> _
	Function put_UseFixedUDPPort(<MarshalAs(UnmanagedType.VariantBool)> UseFixedUDPPort As Boolean) As Integer

	<PreserveSig> _
	Function get_FixedUDPPort(ByRef pFixedUDPPort As Integer) As Integer

	<PreserveSig> _
	Function put_FixedUDPPort(FixedUDPPort As Integer) As Integer

	<PreserveSig> _
	Function get_UseHTTPProxy(<MarshalAs(UnmanagedType.VariantBool)> ByRef pUseHTTPProxy As Boolean) As Integer

	<PreserveSig> _
	Function put_UseHTTPProxy(<MarshalAs(UnmanagedType.VariantBool)> UseHTTPProxy As Boolean) As Integer

	<PreserveSig> _
	Function get_EnableAutoProxy(<MarshalAs(UnmanagedType.VariantBool)> ByRef pEnableAutoProxy As Boolean) As Integer

	<PreserveSig> _
	Function put_EnableAutoProxy(<MarshalAs(UnmanagedType.VariantBool)> EnableAutoProxy As Boolean) As Integer

	<PreserveSig> _
	Function get_HTTPProxyHost(<MarshalAs(UnmanagedType.BStr)> ByRef pbstrHTTPProxyHost As String) As Integer

	<PreserveSig> _
	Function put_HTTPProxyHost(<MarshalAs(UnmanagedType.BStr)> bstrHTTPProxyHost As String) As Integer

	<PreserveSig> _
	Function get_HTTPProxyPort(ByRef pHTTPProxyPort As Integer) As Integer

	<PreserveSig> _
	Function put_HTTPProxyPort(HTTPProxyPort As Integer) As Integer

	<PreserveSig> _
	Function get_EnableMulticast(<MarshalAs(UnmanagedType.VariantBool)> ByRef pEnableMulticast As Boolean) As Integer

	<PreserveSig> _
	Function put_EnableMulticast(<MarshalAs(UnmanagedType.VariantBool)> EnableMulticast As Boolean) As Integer

	<PreserveSig> _
	Function get_EnableUDP(<MarshalAs(UnmanagedType.VariantBool)> ByRef pEnableUDP As Boolean) As Integer

	<PreserveSig> _
	Function put_EnableUDP(<MarshalAs(UnmanagedType.VariantBool)> EnableUDP As Boolean) As Integer

	<PreserveSig> _
	Function get_EnableTCP(<MarshalAs(UnmanagedType.VariantBool)> ByRef pEnableTCP As Boolean) As Integer

	<PreserveSig> _
	Function put_EnableTCP(<MarshalAs(UnmanagedType.VariantBool)> EnableTCP As Boolean) As Integer

	<PreserveSig> _
	Function get_EnableHTTP(<MarshalAs(UnmanagedType.VariantBool)> ByRef pEnableHTTP As Boolean) As Integer

	<PreserveSig> _
	Function put_EnableHTTP(<MarshalAs(UnmanagedType.VariantBool)> EnableHTTP As Boolean) As Integer

End Interface

<ComImport, System.Security.SuppressUnmanagedCodeSecurity, Guid("FA2AA8F2-8B62-11D0-A520-000000000000"), InterfaceType(ComInterfaceType.InterfaceIsDual)> _
Public Interface IAMChannelInfo
	<PreserveSig> _
	Function get_ChannelName(<MarshalAs(UnmanagedType.BStr)> ByRef pbstrChannelName As String) As Integer

	<PreserveSig> _
	Function get_ChannelDescription(<MarshalAs(UnmanagedType.BStr)> ByRef pbstrChannelDescription As String) As Integer

	<PreserveSig> _
	Function get_ChannelURL(<MarshalAs(UnmanagedType.BStr)> ByRef pbstrChannelURL As String) As Integer

	<PreserveSig> _
	Function get_ContactAddress(<MarshalAs(UnmanagedType.BStr)> ByRef pbstrContactAddress As String) As Integer

	<PreserveSig> _
	Function get_ContactPhone(<MarshalAs(UnmanagedType.BStr)> ByRef pbstrContactPhone As String) As Integer

	<PreserveSig> _
	Function get_ContactEmail(<MarshalAs(UnmanagedType.BStr)> ByRef pbstrContactEmail As String) As Integer

End Interface

<ComImport, System.Security.SuppressUnmanagedCodeSecurity, Guid("FA2AA8F3-8B62-11D0-A520-000000000000"), InterfaceType(ComInterfaceType.InterfaceIsDual)> _
Public Interface IAMNetworkStatus
	<PreserveSig> _
	Function get_ReceivedPackets(ByRef pReceivedPackets As Integer) As Integer

	<PreserveSig> _
	Function get_RecoveredPackets(ByRef pRecoveredPackets As Integer) As Integer

	<PreserveSig> _
	Function get_LostPackets(ByRef pLostPackets As Integer) As Integer

	<PreserveSig> _
	Function get_ReceptionQuality(ByRef pReceptionQuality As Integer) As Integer

	<PreserveSig> _
	Function get_BufferingCount(ByRef pBufferingCount As Integer) As Integer

	<PreserveSig> _
	Function get_IsBroadcast(<MarshalAs(UnmanagedType.VariantBool)> ByRef pIsBroadcast As Boolean) As Integer

	<PreserveSig> _
	Function get_BufferingProgress(ByRef pBufferingProgress As Integer) As Integer

End Interface

<ComImport, System.Security.SuppressUnmanagedCodeSecurity, Guid("FA2AA8F5-8B62-11D0-A520-000000000000"), InterfaceType(ComInterfaceType.InterfaceIsDual)> _
Public Interface IAMNetShowExProps
	<PreserveSig> _
	Function get_SourceProtocol(ByRef pSourceProtocol As Integer) As Integer

	<PreserveSig> _
	Function get_Bandwidth(ByRef pBandwidth As Integer) As Integer

	<PreserveSig> _
	Function get_ErrorCorrection(<MarshalAs(UnmanagedType.BStr)> ByRef pbstrErrorCorrection As String) As Integer

	<PreserveSig> _
	Function get_CodecCount(ByRef pCodecCount As Integer) As Integer

	<PreserveSig> _
	Function GetCodecInstalled(CodecNum As Integer, <MarshalAs(UnmanagedType.VariantBool)> ByRef pCodecInstalled As Boolean) As Integer

	<PreserveSig> _
	Function GetCodecDescription(CodecNum As Integer, <MarshalAs(UnmanagedType.BStr)> ByRef pbstrCodecDescription As String) As Integer

	<PreserveSig> _
	Function GetCodecURL(CodecNum As Integer, <MarshalAs(UnmanagedType.BStr)> ByRef pbstrCodecURL As String) As Integer

	<PreserveSig> _
	Function get_CreationDate(ByRef pCreationDate As Double) As Integer

	<PreserveSig> _
	Function get_SourceLink(<MarshalAs(UnmanagedType.BStr)> ByRef pbstrSourceLink As String) As Integer

End Interface

<ComImport, System.Security.SuppressUnmanagedCodeSecurity, Guid("FA2AA8F6-8B62-11D0-A520-000000000000"), InterfaceType(ComInterfaceType.InterfaceIsDual)> _
Public Interface IAMExtendedErrorInfo
	<PreserveSig> _
	Function get_HasError(<MarshalAs(UnmanagedType.VariantBool)> ByRef pHasError As Boolean) As Integer

	<PreserveSig> _
	Function get_ErrorDescription(<MarshalAs(UnmanagedType.BStr)> ByRef pbstrErrorDescription As String) As Integer

	<PreserveSig> _
	Function get_ErrorCode(ByRef pErrorCode As Integer) As Integer

End Interface

<ComImport, System.Security.SuppressUnmanagedCodeSecurity, Guid("AAE7E4E2-6388-11D1-8D93-006097C9A2B2"), InterfaceType(ComInterfaceType.InterfaceIsDual)> _
Public Interface IAMNetShowPreroll
	<PreserveSig> _
	Function put_Preroll(<MarshalAs(UnmanagedType.VariantBool)> fPreroll As Boolean) As Integer

	<PreserveSig> _
	Function get_Preroll(<MarshalAs(UnmanagedType.VariantBool)> ByRef pfPreroll As Boolean) As Integer

End Interface

<ComImport, System.Security.SuppressUnmanagedCodeSecurity, Guid("4746B7C8-700E-11D1-BECC-00C04FB6E937"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)> _
Public Interface IDShowPlugin
	<PreserveSig> _
	Function get_URL(<MarshalAs(UnmanagedType.BStr)> ByRef pURL As String) As Integer

	<PreserveSig> _
	Function get_UserAgent(<MarshalAs(UnmanagedType.BStr)> ByRef pUserAgent As String) As Integer

End Interface

#End If

<ComImport, System.Security.SuppressUnmanagedCodeSecurity, Guid("FA2AA8F4-8B62-11D0-A520-000000000000"), InterfaceType(ComInterfaceType.InterfaceIsDual)> _
Public Interface IAMMediaContent
	<PreserveSig> _
	Function get_AuthorName(<MarshalAs(UnmanagedType.BStr)> ByRef pbstrAuthorName As String) As Integer

	<PreserveSig> _
	Function get_Title(<MarshalAs(UnmanagedType.BStr)> ByRef pbstrTitle As String) As Integer

	<PreserveSig> _
	Function get_Rating(<MarshalAs(UnmanagedType.BStr)> ByRef pbstrRating As String) As Integer

	<PreserveSig> _
	Function get_Description(<MarshalAs(UnmanagedType.BStr)> ByRef pbstrDescription As String) As Integer

	<PreserveSig> _
	Function get_Copyright(<MarshalAs(UnmanagedType.BStr)> ByRef pbstrCopyright As String) As Integer

	<PreserveSig> _
	Function get_BaseURL(<MarshalAs(UnmanagedType.BStr)> ByRef pbstrBaseURL As String) As Integer

	<PreserveSig> _
	Function get_LogoURL(<MarshalAs(UnmanagedType.BStr)> ByRef pbstrLogoURL As String) As Integer

	<PreserveSig> _
	Function get_LogoIconURL(<MarshalAs(UnmanagedType.BStr)> ByRef pbstrLogoURL As String) As Integer

	<PreserveSig> _
	Function get_WatermarkURL(<MarshalAs(UnmanagedType.BStr)> ByRef pbstrWatermarkURL As String) As Integer

	<PreserveSig> _
	Function get_MoreInfoURL(<MarshalAs(UnmanagedType.BStr)> ByRef pbstrMoreInfoURL As String) As Integer

	<PreserveSig> _
	Function get_MoreInfoBannerImage(<MarshalAs(UnmanagedType.BStr)> ByRef pbstrMoreInfoBannerImage As String) As Integer

	<PreserveSig> _
	Function get_MoreInfoBannerURL(<MarshalAs(UnmanagedType.BStr)> ByRef pbstrMoreInfoBannerURL As String) As Integer

	<PreserveSig> _
	Function get_MoreInfoText(<MarshalAs(UnmanagedType.BStr)> ByRef pbstrMoreInfoText As String) As Integer

End Interface

<ComImport, System.Security.SuppressUnmanagedCodeSecurity, Guid("FA2AA8F9-8B62-11D0-A520-000000000000"), InterfaceType(ComInterfaceType.InterfaceIsDual)> _
Public Interface IAMExtendedSeeking
	<PreserveSig> _
	Function get_ExSeekCapabilities(ByRef pExCapabilities As AMExtendedSeekingCapabilities) As Integer

	<PreserveSig> _
	Function get_MarkerCount(ByRef pMarkerCount As Integer) As Integer

	<PreserveSig> _
	Function get_CurrentMarker(ByRef pCurrentMarker As Integer) As Integer

	<PreserveSig> _
	Function GetMarkerTime(MarkerNum As Integer, ByRef pMarkerTime As Double) As Integer

	<PreserveSig> _
	Function GetMarkerName(MarkerNum As Integer, <MarshalAs(UnmanagedType.BStr)> ByRef pbstrMarkerName As String) As Integer

	<PreserveSig> _
	Function put_PlaybackSpeed(Speed As Double) As Integer

	<PreserveSig> _
	Function get_PlaybackSpeed(ByRef pSpeed As Double) As Integer

End Interface

<ComImport, System.Security.SuppressUnmanagedCodeSecurity, Guid("CE8F78C1-74D9-11D2-B09D-00A0C9A81117"), InterfaceType(ComInterfaceType.InterfaceIsDual)> _
Public Interface IAMMediaContent2
	<PreserveSig> _
	Function get_MediaParameter(EntryNum As Integer, <MarshalAs(UnmanagedType.BStr)> bstrName As String, <MarshalAs(UnmanagedType.BStr)> ByRef pbstrValue As String) As Integer

	<PreserveSig> _
	Function get_MediaParameterName(EntryNum As Integer, Index As Integer, <MarshalAs(UnmanagedType.BStr)> ByRef pbstrName As String) As Integer

	<PreserveSig> _
	Function get_PlaylistCount(ByRef pNumberEntries As Integer) As Integer

End Interface

