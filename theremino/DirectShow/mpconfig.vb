
Imports System.Runtime.InteropServices

<Flags> _
Public Enum DDColor
	None = &H0
	Brightness = &H1
	Contrast = &H2
	Hue = &H4
	Saturation = &H8
	Sharpness = &H10
	Gamma = &H20
	ColorEnable = &H40
End Enum

<StructLayout(LayoutKind.Sequential)> _
Public Class DDColorControl
	Public dwSize As Integer
	Public dwFlags As DDColor
	Public lBrightness As Integer
	Public lContrast As Integer
	Public lHue As Integer
	Public lSaturation As Integer
	Public lSharpness As Integer
	Public lGamma As Integer
	Public lColorEnable As Integer
	Public dwReserved1 As Integer
End Class

Public Enum AspectRatioMode
	Stretched
	LetterBox
	Crop
	StretchedAsPrimary
End Enum


<ComImport, System.Security.SuppressUnmanagedCodeSecurity, Guid("EBF47182-8764-11d1-9E69-00C04FD7C15B"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)> _
Public Interface IMixerPinConfig2
	Inherits IMixerPinConfig

    <PreserveSig()> _
    Shadows Function SetRelativePosition(ByVal dwLeft As Integer, ByVal dwTop As Integer, ByVal dwRight As Integer, ByVal dwBottom As Integer) As Integer

    <PreserveSig()> _
    Shadows Function GetRelativePosition(ByRef pdwLeft As Integer, ByRef pdwTop As Integer, ByRef pdwRight As Integer, ByRef pdwBottom As Integer) As Integer

    <PreserveSig()> _
    Shadows Function SetZOrder(ByVal dwZOrder As Integer) As Integer

    <PreserveSig()> _
    Shadows Function GetZOrder(ByRef pdwZOrder As Integer) As Integer

    <PreserveSig()> _
    Shadows Function SetColorKey(<MarshalAs(UnmanagedType.LPStruct)> ByVal pColorKey As ColorKey) As Integer

    <PreserveSig()> _
    Shadows Function GetColorKey(<Out(), MarshalAs(UnmanagedType.LPStruct)> ByVal pColorKey As ColorKey, ByRef pColor As Integer) As Integer

    <PreserveSig()> _
    Shadows Function SetBlendingParameter(ByVal dwBlendingParameter As Integer) As Integer

    <PreserveSig()> _
    Shadows Function GetBlendingParameter(ByRef pdwBlendingParameter As Integer) As Integer

    <PreserveSig()> _
    Shadows Function SetAspectRatioMode(ByVal amAspectRatioMode As AspectRatioMode) As Integer

    <PreserveSig()> _
    Shadows Function GetAspectRatioMode(ByRef pamAspectRatioMode As AspectRatioMode) As Integer

    <PreserveSig()> _
    Shadows Function SetStreamTransparent(<[In](), MarshalAs(UnmanagedType.Bool)> ByVal bStreamTransparent As Boolean) As Integer

    <PreserveSig()> _
    Shadows Function GetStreamTransparent(<Out(), MarshalAs(UnmanagedType.Bool)> ByRef pbStreamTransparent As Boolean) As Integer

	<PreserveSig> _
	Function SetOverlaySurfaceColorControls(pColorControl As DDColorControl) As Integer

	<PreserveSig> _
	Function GetOverlaySurfaceColorControls(pColorControl As DDColorControl) As Integer
End Interface

<ComImport, System.Security.SuppressUnmanagedCodeSecurity, Guid("593CDDE1-0759-11d1-9E69-00C04FD7C15B"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)> _
Public Interface IMixerPinConfig
	<PreserveSig> _
	Function SetRelativePosition(dwLeft As Integer, dwTop As Integer, dwRight As Integer, dwBottom As Integer) As Integer

	<PreserveSig> _
	Function GetRelativePosition(ByRef pdwLeft As Integer, ByRef pdwTop As Integer, ByRef pdwRight As Integer, ByRef pdwBottom As Integer) As Integer

	<PreserveSig> _
	Function SetZOrder(dwZOrder As Integer) As Integer

	<PreserveSig> _
	Function GetZOrder(ByRef pdwZOrder As Integer) As Integer

	<PreserveSig> _
	Function SetColorKey(<MarshalAs(UnmanagedType.LPStruct)> pColorKey As ColorKey) As Integer

	<PreserveSig> _
	Function GetColorKey(<Out, MarshalAs(UnmanagedType.LPStruct)> pColorKey As ColorKey, ByRef pColor As Integer) As Integer

	<PreserveSig> _
	Function SetBlendingParameter(dwBlendingParameter As Integer) As Integer

	<PreserveSig> _
	Function GetBlendingParameter(ByRef pdwBlendingParameter As Integer) As Integer

	<PreserveSig> _
	Function SetAspectRatioMode(amAspectRatioMode As AspectRatioMode) As Integer

	<PreserveSig> _
	Function GetAspectRatioMode(ByRef pamAspectRatioMode As AspectRatioMode) As Integer

	<PreserveSig> _
	Function SetStreamTransparent(<[In], MarshalAs(UnmanagedType.Bool)> bStreamTransparent As Boolean) As Integer

	<PreserveSig> _
	Function GetStreamTransparent(<Out, MarshalAs(UnmanagedType.Bool)> ByRef pbStreamTransparent As Boolean) As Integer
End Interface

