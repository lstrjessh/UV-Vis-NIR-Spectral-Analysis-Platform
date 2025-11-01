
Imports System.Runtime.InteropServices

Public Enum AMLine21CCLevel
	TC2 = 0
End Enum

Public Enum AMLine21CCService
	None = 0
	Caption1
	Caption2
	Text1
	Text2
	XDS
	DefChannel = 10
	Invalid
End Enum

Public Enum AMLine21CCState
	Off = 0
	[On]
End Enum

Public Enum AMLine21DrawBGMode
	Opaque
	Transparent
End Enum


<ComImport, System.Security.SuppressUnmanagedCodeSecurity, Guid("6E8D4A21-310C-11d0-B79A-00AA003767A7"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)> _
Public Interface IAMLine21Decoder
	<PreserveSig> _
	Function GetDecoderLevel(<Out> ByRef lpLevel As AMLine21CCLevel) As Integer

	<PreserveSig> _
	Function GetCurrentService(<Out> ByRef lpService As AMLine21CCService) As Integer

	<PreserveSig> _
	Function SetCurrentService(<[In]> Service As AMLine21CCService) As Integer

	<PreserveSig> _
	Function GetServiceState(<Out> ByRef lpState As AMLine21CCState) As Integer

	<PreserveSig> _
	Function SetServiceState(<[In]> State As AMLine21CCState) As Integer

	<PreserveSig> _
	Function GetOutputFormat(<Out> lpbmih As BitmapInfoHeader) As Integer

	<PreserveSig> _
	Function SetOutputFormat(<[In]> lpbmih As BitmapInfoHeader) As Integer

	<PreserveSig> _
	Function GetBackgroundColor(<Out> ByRef pdwPhysColor As Integer) As Integer

	<PreserveSig> _
	Function SetBackgroundColor(<[In]> dwPhysColor As Integer) As Integer

	<PreserveSig> _
	Function GetRedrawAlways(<Out, MarshalAs(UnmanagedType.Bool)> ByRef lpbOption As Boolean) As Integer

	<PreserveSig> _
	Function SetRedrawAlways(<[In], MarshalAs(UnmanagedType.Bool)> bOption As Boolean) As Integer

	<PreserveSig> _
	Function GetDrawBackgroundMode(<Out> ByRef lpMode As AMLine21DrawBGMode) As Integer

	<PreserveSig> _
	Function SetDrawBackgroundMode(<[In]> Mode As AMLine21DrawBGMode) As Integer
End Interface

