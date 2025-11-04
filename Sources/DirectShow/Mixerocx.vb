
Imports System.Runtime.InteropServices

<Flags> _
Public Enum MixerData
	None = 0
    AspectRatio = &H1   ' picture aspect ratio changed
    NativeSize = &H2    ' native size of video changed
    Palette = &H4       ' palette of video changed
End Enum

Public Enum MixerState
    Mask = &H3              ' use this mask with state status bits
    Unconnected = &H0       ' mixer is unconnected and stopped
    ConnectedStopped = &H1  ' mixer is connected and stopped
    ConnectedPaused = &H2   ' mixer is connected and paused
    ConnectedPlaying = &H3  ' mixer is connected and playing
End Enum


<ComImport, System.Security.SuppressUnmanagedCodeSecurity, Guid("81A3BD31-DEE1-11d1-8508-00A0C91F9CA0"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)> _
Public Interface IMixerOCXNotify
	<PreserveSig> _
	Function OnInvalidateRect(<[In]> lpcRect As DsRect) As Integer

	<PreserveSig> _
	Function OnStatusChange(<[In]> ulStatusFlags As MixerState) As Integer

	<PreserveSig> _
	Function OnDataChange(<[In]> ulDataFlags As MixerData) As Integer
End Interface

<ComImport, System.Security.SuppressUnmanagedCodeSecurity, Guid("81A3BD32-DEE1-11d1-8508-00A0C91F9CA0"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)> _
Public Interface IMixerOCX
	<PreserveSig> _
	Function OnDisplayChange(<[In]> ulBitsPerPixel As Integer, <[In]> ulScreenWidth As Integer, <[In]> ulScreenHeight As Integer) As Integer

	<PreserveSig> _
	Function GetAspectRatio(<Out> ByRef pdwPictAspectRatioX As Integer, <Out> ByRef pdwPictAspectRatioY As Integer) As Integer

	<PreserveSig> _
	Function GetVideoSize(<Out> ByRef pdwVideoWidth As Integer, <Out> ByRef pdwVideoHeight As Integer) As Integer

	<PreserveSig> _
	Function GetStatus(<Out> ByRef pdwStatus As Integer) As Integer

	' HDC
	<PreserveSig> _
	Function OnDraw(<[In]> hdcDraw As IntPtr, <[In]> prcDraw As DsRect) As Integer

	' While in theory this takes an LPPOINT, in practice
	' it must be NULL.
	<PreserveSig> _
	Function SetDrawRegion(<[In]> lpptTopLeftSC As IntPtr, <[In]> prcDrawCC As DsRect, <[In]> lprcClip As DsRect) As Integer

	<PreserveSig> _
	Function Advise(<[In]> pmdns As IMixerOCXNotify) As Integer

	<PreserveSig> _
	Function UnAdvise() As Integer
End Interface

