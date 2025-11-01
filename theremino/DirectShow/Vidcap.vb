
Imports System.Runtime.InteropServices

<StructLayout(LayoutKind.Sequential)> _
Public Structure KSTopologyConnection
	Public FromNode As Integer
	Public FromNodePin As Integer
	Public ToNode As Integer
	Public ToNodePin As Integer
End Structure



#If ALLOW_UNTESTED_INTERFACES Then

<ComImport, System.Security.SuppressUnmanagedCodeSecurity, InterfaceType(ComInterfaceType.InterfaceIsIUnknown), Guid("4050560E-42A7-413a-85C2-09269A2D0F44")> _
Public Interface IVideoProcAmp
	<PreserveSig> _
	Function get_BacklightCompensation(ByRef pValue As Integer, ByRef pFlags As VideoProcAmpFlags) As Integer

	<PreserveSig> _
	Function put_BacklightCompensation(Value As Integer, Flags As VideoProcAmpFlags) As Integer

	<PreserveSig> _
	Function getRange_BacklightCompensation(ByRef pMin As Integer, ByRef pMax As Integer, ByRef pSteppingDelta As Integer, ByRef pDefault As Integer, ByRef pCapsFlag As VideoProcAmpFlags) As Integer

	<PreserveSig> _
	Function get_Brightness(ByRef pValue As Integer, ByRef pFlags As VideoProcAmpFlags) As Integer

	<PreserveSig> _
	Function put_Brightness(Value As Integer, Flags As VideoProcAmpFlags) As Integer

	<PreserveSig> _
	Function getRange_Brightness(ByRef pMin As Integer, ByRef pMax As Integer, ByRef pSteppingDelta As Integer, ByRef pDefault As Integer, ByRef pCapsFlag As VideoProcAmpFlags) As Integer

	<PreserveSig> _
	Function get_ColorEnable(ByRef pValue As Integer, ByRef pFlags As VideoProcAmpFlags) As Integer

	<PreserveSig> _
	Function put_ColorEnable(Value As Integer, Flags As VideoProcAmpFlags) As Integer

	<PreserveSig> _
	Function getRange_ColorEnable(ByRef pMin As Integer, ByRef pMax As Integer, ByRef pSteppingDelta As Integer, ByRef pDefault As Integer, ByRef pCapsFlag As VideoProcAmpFlags) As Integer

	<PreserveSig> _
	Function get_Contrast(ByRef pValue As Integer, ByRef pFlags As VideoProcAmpFlags) As Integer

	<PreserveSig> _
	Function put_Contrast(Value As Integer, Flags As VideoProcAmpFlags) As Integer

	<PreserveSig> _
	Function getRange_Contrast(ByRef pMin As Integer, ByRef pMax As Integer, ByRef pSteppingDelta As Integer, ByRef pDefault As Integer, ByRef pCapsFlag As VideoProcAmpFlags) As Integer

	<PreserveSig> _
	Function get_Gamma(ByRef pValue As Integer, ByRef pFlags As VideoProcAmpFlags) As Integer

	<PreserveSig> _
	Function put_Gamma(Value As Integer, Flags As VideoProcAmpFlags) As Integer

	<PreserveSig> _
	Function getRange_Gamma(ByRef pMin As Integer, ByRef pMax As Integer, ByRef pSteppingDelta As Integer, ByRef pDefault As Integer, ByRef pCapsFlag As VideoProcAmpFlags) As Integer

	<PreserveSig> _
	Function get_Saturation(ByRef pValue As Integer, ByRef pFlags As VideoProcAmpFlags) As Integer

	<PreserveSig> _
	Function put_Saturation(Value As Integer, Flags As VideoProcAmpFlags) As Integer

	<PreserveSig> _
	Function getRange_Saturation(ByRef pMin As Integer, ByRef pMax As Integer, ByRef pSteppingDelta As Integer, ByRef pDefault As Integer, ByRef pCapsFlag As VideoProcAmpFlags) As Integer

	<PreserveSig> _
	Function get_Sharpness(ByRef pValue As Integer, ByRef pFlags As VideoProcAmpFlags) As Integer

	<PreserveSig> _
	Function put_Sharpness(Value As Integer, Flags As VideoProcAmpFlags) As Integer

	<PreserveSig> _
	Function getRange_Sharpness(ByRef pMin As Integer, ByRef pMax As Integer, ByRef pSteppingDelta As Integer, ByRef pDefault As Integer, ByRef pCapsFlag As VideoProcAmpFlags) As Integer

	<PreserveSig> _
	Function get_WhiteBalance(ByRef pValue As Integer, ByRef pFlags As VideoProcAmpFlags) As Integer

	<PreserveSig> _
	Function put_WhiteBalance(Value As Integer, Flags As VideoProcAmpFlags) As Integer

	<PreserveSig> _
	Function getRange_WhiteBalance(ByRef pMin As Integer, ByRef pMax As Integer, ByRef pSteppingDelta As Integer, ByRef pDefault As Integer, ByRef pCapsFlag As VideoProcAmpFlags) As Integer

	<PreserveSig> _
	Function get_Gain(ByRef pValue As Integer, ByRef pFlags As VideoProcAmpFlags) As Integer

	<PreserveSig> _
	Function put_Gain(Value As Integer, Flags As VideoProcAmpFlags) As Integer

	<PreserveSig> _
	Function getRange_Gain(ByRef pMin As Integer, ByRef pMax As Integer, ByRef pSteppingDelta As Integer, ByRef pDefault As Integer, ByRef pCapsFlag As VideoProcAmpFlags) As Integer

	<PreserveSig> _
	Function get_Hue(ByRef pValue As Integer, ByRef pFlags As VideoProcAmpFlags) As Integer

	<PreserveSig> _
	Function put_Hue(Value As Integer, Flags As VideoProcAmpFlags) As Integer

	<PreserveSig> _
	Function getRange_Hue(ByRef pMin As Integer, ByRef pMax As Integer, ByRef pSteppingDelta As Integer, ByRef pDefault As Integer, ByRef pCapsFlag As VideoProcAmpFlags) As Integer

	<PreserveSig> _
	Function get_DigitalMultiplier(ByRef pValue As Integer, ByRef pFlags As VideoProcAmpFlags) As Integer

	<PreserveSig> _
	Function put_DigitalMultiplier(Value As Integer, Flags As VideoProcAmpFlags) As Integer

	<PreserveSig> _
	Function getRange_DigitalMultiplier(ByRef pMin As Integer, ByRef pMax As Integer, ByRef pSteppingDelta As Integer, ByRef pDefault As Integer, ByRef pCapsFlag As VideoProcAmpFlags) As Integer

	<PreserveSig> _
	Function get_PowerlineFrequency(ByRef pValue As Integer, ByRef pFlags As VideoProcAmpFlags) As Integer

	<PreserveSig> _
	Function put_PowerlineFrequency(Value As Integer, Flags As VideoProcAmpFlags) As Integer

	<PreserveSig> _
	Function getRange_PowerlineFrequency(ByRef pMin As Integer, ByRef pMax As Integer, ByRef pSteppingDelta As Integer, ByRef pDefault As Integer, ByRef pCapsFlag As VideoProcAmpFlags) As Integer

	<PreserveSig> _
	Function get_WhiteBalanceComponent(ByRef pValue1 As Integer, ByRef pValue2 As Integer, ByRef pFlags As VideoProcAmpFlags) As Integer

	<PreserveSig> _
	Function put_WhiteBalanceComponent(Value1 As Integer, Value2 As Integer, Flags As VideoProcAmpFlags) As Integer

	<PreserveSig> _
	Function getRange_WhiteBalanceComponent(ByRef pMin As Integer, ByRef pMax As Integer, ByRef pSteppingDelta As Integer, ByRef pDefault As Integer, ByRef pCapsFlag As VideoProcAmpFlags) As Integer
End Interface

<ComImport, System.Security.SuppressUnmanagedCodeSecurity, InterfaceType(ComInterfaceType.InterfaceIsIUnknown), Guid("2BA1785D-4D1B-44EF-85E8-C7F1D3F20184")> _
Public Interface ICameraControl
	<PreserveSig> _
	Function get_Exposure(ByRef pValue As Integer, ByRef pFlags As Integer) As Integer

	<PreserveSig> _
	Function put_Exposure(<[In]> Value As Integer, <[In]> Flags As Integer) As Integer

	<PreserveSig> _
	Function getRange_Exposure(ByRef pMin As Integer, ByRef pMax As Integer, ByRef pSteppingDelta As Integer, ByRef pDefault As Integer, ByRef pCapsFlag As Integer) As Integer

	<PreserveSig> _
	Function get_Focus(ByRef pValue As Integer, ByRef pFlags As Integer) As Integer

	<PreserveSig> _
	Function put_Focus(<[In]> Value As Integer, <[In]> Flags As Integer) As Integer

	<PreserveSig> _
	Function getRange_Focus(ByRef pMin As Integer, ByRef pMax As Integer, ByRef pSteppingDelta As Integer, ByRef pDefault As Integer, ByRef pCapsFlag As Integer) As Integer

	<PreserveSig> _
	Function get_Iris(ByRef pValue As Integer, ByRef pFlags As Integer) As Integer

	<PreserveSig> _
	Function put_Iris(<[In]> Value As Integer, <[In]> Flags As Integer) As Integer

	<PreserveSig> _
	Function getRange_Iris(ByRef pMin As Integer, ByRef pMax As Integer, ByRef pSteppingDelta As Integer, ByRef pDefault As Integer, ByRef pCapsFlag As Integer) As Integer

	<PreserveSig> _
	Function get_Zoom(ByRef pValue As Integer, ByRef pFlags As Integer) As Integer

	<PreserveSig> _
	Function put_Zoom(<[In]> Value As Integer, <[In]> Flags As Integer) As Integer

	<PreserveSig> _
	Function getRange_Zoom(ByRef pMin As Integer, ByRef pMax As Integer, ByRef pSteppingDelta As Integer, ByRef pDefault As Integer, ByRef pCapsFlag As Integer) As Integer

	<PreserveSig> _
	Function get_FocalLengths(ByRef plOcularFocalLength As Integer, ByRef plObjectiveFocalLengthMin As Integer, ByRef plObjectiveFocalLengthMax As Integer) As Integer

	<PreserveSig> _
	Function get_Pan(ByRef pValue As Integer, ByRef pFlags As Integer) As Integer

	<PreserveSig> _
	Function put_Pan(<[In]> Value As Integer, <[In]> Flags As Integer) As Integer

	<PreserveSig> _
	Function getRange_Pan(ByRef pMin As Integer, ByRef pMax As Integer, ByRef pSteppingDelta As Integer, ByRef pDefault As Integer, ByRef pCapsFlag As Integer) As Integer

	<PreserveSig> _
	Function get_Tilt(ByRef pValue As Integer, ByRef pFlags As Integer) As Integer

	<PreserveSig> _
	Function put_Tilt(<[In]> Value As Integer, <[In]> Flags As Integer) As Integer

	<PreserveSig> _
	Function getRange_Tilt(ByRef pMin As Integer, ByRef pMax As Integer, ByRef pSteppingDelta As Integer, ByRef pDefault As Integer, ByRef pCapsFlag As Integer) As Integer

	<PreserveSig> _
	Function get_PanTilt(ByRef pPanValue As Integer, ByRef pTiltValue As Integer, ByRef pFlags As Integer) As Integer

	<PreserveSig> _
	Function put_PanTilt(<[In]> PanValue As Integer, <[In]> TiltValue As Integer, <[In]> Flags As Integer) As Integer

	<PreserveSig> _
	Function get_Roll(ByRef pValue As Integer, ByRef pFlags As Integer) As Integer

	<PreserveSig> _
	Function put_Roll(<[In]> Value As Integer, <[In]> Flags As Integer) As Integer

	<PreserveSig> _
	Function getRange_Roll(ByRef pMin As Integer, ByRef pMax As Integer, ByRef pSteppingDelta As Integer, ByRef pDefault As Integer, ByRef pCapsFlag As Integer) As Integer

	<PreserveSig> _
	Function get_ExposureRelative(ByRef pValue As Integer, ByRef pFlags As Integer) As Integer

	<PreserveSig> _
	Function put_ExposureRelative(<[In]> Value As Integer, <[In]> Flags As Integer) As Integer

	<PreserveSig> _
	Function getRange_ExposureRelative(ByRef pMin As Integer, ByRef pMax As Integer, ByRef pSteppingDelta As Integer, ByRef pDefault As Integer, ByRef pCapsFlag As Integer) As Integer

	<PreserveSig> _
	Function get_FocusRelative(ByRef pValue As Integer, ByRef pFlags As Integer) As Integer

	<PreserveSig> _
	Function put_FocusRelative(<[In]> Value As Integer, <[In]> Flags As Integer) As Integer

	<PreserveSig> _
	Function getRange_FocusRelative(ByRef pMin As Integer, ByRef pMax As Integer, ByRef pSteppingDelta As Integer, ByRef pDefault As Integer, ByRef pCapsFlag As Integer) As Integer

	<PreserveSig> _
	Function get_IrisRelative(ByRef pValue As Integer, ByRef pFlags As Integer) As Integer

	<PreserveSig> _
	Function put_IrisRelative(<[In]> Value As Integer, <[In]> Flags As Integer) As Integer

	<PreserveSig> _
	Function getRange_IrisRelative(ByRef pMin As Integer, ByRef pMax As Integer, ByRef pSteppingDelta As Integer, ByRef pDefault As Integer, ByRef pCapsFlag As Integer) As Integer

	<PreserveSig> _
	Function get_ZoomRelative(ByRef pValue As Integer, ByRef pFlags As Integer) As Integer

	<PreserveSig> _
	Function put_ZoomRelative(<[In]> Value As Integer, <[In]> Flags As Integer) As Integer

	<PreserveSig> _
	Function getRange_ZoomRelative(ByRef pMin As Integer, ByRef pMax As Integer, ByRef pSteppingDelta As Integer, ByRef pDefault As Integer, ByRef pCapsFlag As Integer) As Integer

	<PreserveSig> _
	Function get_PanRelative(ByRef pValue As Integer, ByRef pFlags As Integer) As Integer

	<PreserveSig> _
	Function put_PanRelative(<[In]> Value As Integer, <[In]> Flags As Integer) As Integer

	<PreserveSig> _
	Function get_TiltRelative(ByRef pValue As Integer, ByRef pFlags As Integer) As Integer

	<PreserveSig> _
	Function put_TiltRelative(<[In]> Value As Integer, <[In]> Flags As Integer) As Integer

	<PreserveSig> _
	Function getRange_TiltRelative(ByRef pMin As Integer, ByRef pMax As Integer, ByRef pSteppingDelta As Integer, ByRef pDefault As Integer, ByRef pCapsFlag As Integer) As Integer

	<PreserveSig> _
	Function get_PanTiltRelative(ByRef pPanValue As Integer, ByRef pTiltValue As Integer, ByRef pFlags As Integer) As Integer

	<PreserveSig> _
	Function put_PanTiltRelative(<[In]> PanValue As Integer, <[In]> TiltValue As Integer, <[In]> Flags As Integer) As Integer

	<PreserveSig> _
	Function getRange_PanRelative(ByRef pMin As Integer, ByRef pMax As Integer, ByRef pSteppingDelta As Integer, ByRef pDefault As Integer, ByRef pCapsFlag As Integer) As Integer

	<PreserveSig> _
	Function get_RollRelative(ByRef pValue As Integer, ByRef pFlags As Integer) As Integer

	<PreserveSig> _
	Function put_RollRelative(<[In]> Value As Integer, <[In]> Flags As Integer) As Integer

	<PreserveSig> _
	Function getRange_RollRelative(ByRef pMin As Integer, ByRef pMax As Integer, ByRef pSteppingDelta As Integer, ByRef pDefault As Integer, ByRef pCapsFlag As Integer) As Integer

	<PreserveSig> _
	Function get_ScanMode(ByRef pValue As Integer, ByRef pFlags As Integer) As Integer

	<PreserveSig> _
	Function put_ScanMode(<[In]> Value As Integer, <[In]> Flags As Integer) As Integer

	<PreserveSig> _
	Function get_PrivacyMode(ByRef pValue As Integer, ByRef pFlags As Integer) As Integer

	<PreserveSig> _
	Function put_PrivacyMode(<[In]> Value As Integer, <[In]> Flags As Integer) As Integer
End Interface

<ComImport, System.Security.SuppressUnmanagedCodeSecurity, Guid("1ABDAECA-68B6-4F83-9371-B413907C7B9F"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)> _
Public Interface ISelector
	<PreserveSig> _
	Function get_NumSources(<Out> ByRef pdwNumSources As Integer) As Integer

	<PreserveSig> _
	Function get_SourceNodeId(<Out> ByRef pdwPinId As Integer) As Integer

	<PreserveSig> _
	Function put_SourceNodeId(<[In]> dwPinId As Integer) As Integer
End Interface

<ComImport, System.Security.SuppressUnmanagedCodeSecurity, Guid("11737C14-24A7-4bb5-81A0-0D003813B0C4"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)> _
Public Interface IKsNodeControl
	<PreserveSig> _
	Function put_NodeId(<[In]> dwNodeId As Integer) As Integer

	<PreserveSig> _
	Function put_KsControl(<[In]> pKsControl As IntPtr) As Integer ' PVOID
End Interface

#End If

<ComImport, System.Security.SuppressUnmanagedCodeSecurity, Guid("720D4AC0-7533-11D0-A5D6-28DB04C10000"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)> _
Public Interface IKsTopologyInfo
	<PreserveSig> _
	Function get_NumCategories(<Out> ByRef pdwNumCategories As Integer) As Integer

	<PreserveSig> _
	Function get_Category(<[In]> dwIndex As Integer, <Out> ByRef pCategory As Guid) As Integer

	<PreserveSig> _
	Function get_NumConnections(<Out> ByRef pdwNumConnections As Integer) As Integer

	<PreserveSig> _
	Function get_ConnectionInfo(<[In]> dwIndex As Integer, <Out> ByRef pConnectionInfo As KSTopologyConnection) As Integer

	<PreserveSig> _
	Function get_NodeName(<[In]> dwNodeId As Integer, <[In]> pwchNodeName As IntPtr, <[In]> dwBufSize As Integer, <Out> ByRef pdwNameLen As Integer) As Integer

	<PreserveSig> _
	Function get_NumNodes(<Out> ByRef pdwNumNodes As Integer) As Integer

	<PreserveSig> _
	Function get_NodeType(<[In]> dwNodeId As Integer, <Out> ByRef pNodeType As Guid) As Integer

	<PreserveSig> _
	Function CreateNodeInstance(<[In]> dwNodeId As Integer, <[In], MarshalAs(UnmanagedType.LPStruct)> iid As Guid, <Out, MarshalAs(UnmanagedType.IUnknown)> ByRef ppvObject As [Object]) As Integer
End Interface

