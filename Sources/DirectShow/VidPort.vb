
Imports System.Drawing
Imports System.Runtime.InteropServices

#If ALLOW_UNTESTED_INTERFACES Then

<StructLayout(LayoutKind.Sequential)> _
Public Structure AMVPSize
	Public dwWidth As Integer
	Public dwHeight As Integer
End Structure

<StructLayout(LayoutKind.Sequential)> _
Public Structure DDVideoPortConnect
	Public dwSize As Integer
	Public dwPortWidth As Integer
	Public guidTypeID As Guid
	Public dwFlags As Integer
	Public dwReserved1 As IntPtr
End Structure

<StructLayout(LayoutKind.Sequential)> _
Public Structure VPDataInfo
	Public dwSize As Integer
	Public dwMicrosecondsPerField As Integer
	Public amvpDimInfo As AMVPDimInfo
	Public dwPictAspectRatioX As Integer
	Public dwPictAspectRatioY As Integer
	Public bEnableDoubleClock As Boolean
	Public bEnableVACT As Boolean
	Public bDataIsInterlaced As Boolean
	Public lHalfLinesOdd As Integer
	Public bFieldPolarityInverted As Boolean
	Public dwNumLinesInVREF As Integer
	Public lHalfLinesEven As Integer
	Public dwReserved1 As Integer
End Structure

<StructLayout(LayoutKind.Sequential)> _
Public Structure AMVPDimInfo
	Public dwFieldWidth As Integer
	Public dwFieldHeight As Integer
	Public dwVBIWidth As Integer
	Public dwVBIHeight As Integer
	Public rcValidRegion As Rectangle
End Structure

#End If


Public Enum AMVP_Mode
	Weave
	BobInterleaved
	BobNonInterleaved
	SkipEven
	SkipOdd
End Enum



#If ALLOW_UNTESTED_INTERFACES Then

<InterfaceType(ComInterfaceType.InterfaceIsIUnknown)> _
Public Interface IVPBaseConfig
	<PreserveSig> _
	Function GetConnectInfo(ByRef pdwNumConnectInfo As Integer, ByRef pddVPConnectInfo As DDVideoPortConnect) As Integer

	<PreserveSig> _
	Function SetConnectInfo(dwChosenEntry As Integer) As Integer

	<PreserveSig> _
	Function GetVPDataInfo(ByRef pamvpDataInfo As VPDataInfo) As Integer

	<PreserveSig> _
	Function GetMaxPixelRate(ByRef pamvpSize As AMVPSize, ByRef pdwMaxPixelsPerSecond As Integer) As Integer

	<PreserveSig> _
	Function InformVPInputFormats(dwNumFormats As Integer, pDDPixelFormats As DDPixelFormat) As Integer

	<PreserveSig> _
	Function GetVideoFormats(ByRef pdwNumFormats As Integer, ByRef pddPixelFormats As DDPixelFormat) As Integer

	<PreserveSig> _
	Function SetVideoFormat(dwChosenEntry As Integer) As Integer

	<PreserveSig> _
	Function SetInvertPolarity() As Integer

	' IDirectDrawSurface
	<PreserveSig> _
	Function GetOverlaySurface(ByRef ppddOverlaySurface As IntPtr) As Integer

	<PreserveSig> _
	Function SetDirectDrawKernelHandle(dwDDKernelHandle As IntPtr) As Integer

	<PreserveSig> _
	Function SetVideoPortID(dwVideoPortID As Integer) As Integer

	<PreserveSig> _
	Function SetDDSurfaceKernelHandles(cHandles As Integer, rgDDKernelHandles As IntPtr) As Integer

	<PreserveSig> _
	Function SetSurfaceParameters(dwPitch As Integer, dwXOrigin As Integer, dwYOrigin As Integer) As Integer
End Interface

<ComImport, System.Security.SuppressUnmanagedCodeSecurity, Guid("BC29A660-30E3-11d0-9E69-00C04FD7C15B"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)> _
Public Interface IVPConfig
	Inherits IVPBaseConfig

	<PreserveSig> _
	Shadows Function GetConnectInfo(ByRef pdwNumConnectInfo As Integer, ByRef pddVPConnectInfo As DDVideoPortConnect) As Integer

	<PreserveSig> _
	Shadows Function SetConnectInfo(dwChosenEntry As Integer) As Integer

	<PreserveSig> _
	Shadows Function GetVPDataInfo(ByRef pamvpDataInfo As VPDataInfo) As Integer

	<PreserveSig> _
	Shadows Function GetMaxPixelRate(ByRef pamvpSize As AMVPSize, ByRef pdwMaxPixelsPerSecond As Integer) As Integer

	<PreserveSig> _
	Shadows Function InformVPInputFormats(dwNumFormats As Integer, pDDPixelFormats As DDPixelFormat) As Integer

	<PreserveSig> _
	Shadows Function GetVideoFormats(ByRef pdwNumFormats As Integer, ByRef pddPixelFormats As DDPixelFormat) As Integer

	<PreserveSig> _
	Shadows Function SetVideoFormat(dwChosenEntry As Integer) As Integer

	<PreserveSig> _
	Shadows Function SetInvertPolarity() As Integer

	' IDirectDrawSurface
	<PreserveSig> _
	Shadows Function GetOverlaySurface(ByRef ppddOverlaySurface As IntPtr) As Integer

	<PreserveSig> _
	Shadows Function SetDirectDrawKernelHandle(dwDDKernelHandle As IntPtr) As Integer

	<PreserveSig> _
	Shadows Function SetVideoPortID(dwVideoPortID As Integer) As Integer

	<PreserveSig> _
	Shadows Function SetDDSurfaceKernelHandles(cHandles As Integer, rgDDKernelHandles As IntPtr) As Integer

	<PreserveSig> _
	Shadows Function SetSurfaceParameters(dwPitch As Integer, dwXOrigin As Integer, dwYOrigin As Integer) As Integer

	<PreserveSig> _
	Function IsVPDecimationAllowed(<MarshalAs(UnmanagedType.Bool)> ByRef pbIsDecimationAllowed As Boolean) As Integer

	<PreserveSig> _
	Function SetScalingFactors(pamvpSize As AMVPSize) As Integer
End Interface

<ComImport, System.Security.SuppressUnmanagedCodeSecurity, Guid("EC529B00-1A1F-11D1-BAD9-00609744111A"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)> _
Public Interface IVPVBIConfig
	Inherits IVPBaseConfig

	<PreserveSig> _
	Shadows Function GetConnectInfo(ByRef pdwNumConnectInfo As Integer, ByRef pddVPConnectInfo As DDVideoPortConnect) As Integer

	<PreserveSig> _
	Shadows Function SetConnectInfo(dwChosenEntry As Integer) As Integer

	<PreserveSig> _
	Shadows Function GetVPDataInfo(ByRef pamvpDataInfo As VPDataInfo) As Integer

	<PreserveSig> _
	Shadows Function GetMaxPixelRate(ByRef pamvpSize As AMVPSize, ByRef pdwMaxPixelsPerSecond As Integer) As Integer

	<PreserveSig> _
	Shadows Function InformVPInputFormats(dwNumFormats As Integer, pDDPixelFormats As DDPixelFormat) As Integer

	<PreserveSig> _
	Shadows Function GetVideoFormats(ByRef pdwNumFormats As Integer, ByRef pddPixelFormats As DDPixelFormat) As Integer

	<PreserveSig> _
	Shadows Function SetVideoFormat(dwChosenEntry As Integer) As Integer

	<PreserveSig> _
	Shadows Function SetInvertPolarity() As Integer

	' IDirectDrawSurface
	<PreserveSig> _
	Shadows Function GetOverlaySurface(ByRef ppddOverlaySurface As IntPtr) As Integer

	<PreserveSig> _
	Shadows Function SetDirectDrawKernelHandle(dwDDKernelHandle As IntPtr) As Integer

	<PreserveSig> _
	Shadows Function SetVideoPortID(dwVideoPortID As Integer) As Integer

	<PreserveSig> _
	Shadows Function SetDDSurfaceKernelHandles(cHandles As Integer, rgDDKernelHandles As IntPtr) As Integer

	<PreserveSig> _
	Shadows Function SetSurfaceParameters(dwPitch As Integer, dwXOrigin As Integer, dwYOrigin As Integer) As Integer

End Interface

<ComImport, System.Security.SuppressUnmanagedCodeSecurity, Guid("EC529B01-1A1F-11D1-BAD9-00609744111A"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)> _
Public Interface IVPVBINotify
	Inherits IVPBaseNotify

	<PreserveSig> _
	Shadows Function RenegotiateVPParameters() As Integer

End Interface

#End If

<InterfaceType(ComInterfaceType.InterfaceIsIUnknown)> _
Public Interface IVPBaseNotify
	<PreserveSig> _
	Function RenegotiateVPParameters() As Integer
End Interface

<ComImport, System.Security.SuppressUnmanagedCodeSecurity, Guid("C76794A1-D6C5-11d0-9E69-00C04FD7C15B"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)> _
Public Interface IVPNotify
	Inherits IVPBaseNotify

    <PreserveSig()> _
    Shadows Function RenegotiateVPParameters() As Integer

	<PreserveSig> _
	Function SetDeinterlaceMode(mode As AMVP_Mode) As Integer

	<PreserveSig> _
	Function GetDeinterlaceMode(ByRef pMode As AMVP_Mode) As Integer
End Interface

<ComImport, System.Security.SuppressUnmanagedCodeSecurity, Guid("EBF47183-8764-11d1-9E69-00C04FD7C15B"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)> _
Public Interface IVPNotify2
	Inherits IVPNotify

    <PreserveSig()> _
    Shadows Function RenegotiateVPParameters() As Integer

    <PreserveSig()> _
    Shadows Function SetDeinterlaceMode(ByVal mode As AMVP_Mode) As Integer

    <PreserveSig()> _
    Shadows Function GetDeinterlaceMode(ByRef pMode As AMVP_Mode) As Integer

	<PreserveSig> _
	Function SetVPSyncMaster(<MarshalAs(UnmanagedType.Bool)> bVPSyncMaster As Boolean) As Integer

	<PreserveSig> _
	Function GetVPSyncMaster(<MarshalAs(UnmanagedType.Bool)> ByRef pbVPSyncMaster As Boolean) As Integer

End Interface

