
Imports System.Drawing
Imports System.Runtime.InteropServices

#If ALLOW_UNTESTED_INTERFACES Then

<Flags> _
Public Enum DirectDrawSwitches
	None = &H0
	DCIPS = &H1
	PS = &H2
	RGBOVR = &H4
	YUVOVR = &H8
	RGBOFF = &H10
	YUVOFF = &H20
	RGBFLP = &H40
	YUVFLP = &H80
	All = &Hff
	YUV = (YUVOFF Or YUVOVR Or YUVFLP)
	RGB = (RGBOFF Or RGBOVR Or RGBFLP)
	Primary = (DCIPS Or PS)
End Enum

Public Enum PropertyFrameStep
	[Step] = &H1
	Cancel = &H2
	CanStep = &H3
	CanStepMultiple = &H4
End Enum

<StructLayout(LayoutKind.Sequential)> _
Public Structure FrameStepStep
	Public dwFramesToStep As Integer
End Structure

<StructLayout(LayoutKind.Sequential)> _
Public Structure MPEG1VideoInfo
	Public hdr As VideoInfoHeader
	Public dwStartTimeCode As Integer
	Public cbSequenceHeader As Integer
	Public bSequenceHeader As Byte
End Structure

<StructLayout(LayoutKind.Sequential)> _
Public Structure AnalogVideoInfo
	Public rcSource As Rectangle
	Public rcTarget As Rectangle
	Public dwActiveWidth As Integer
	Public dwActiveHeight As Integer
	Public AvgTimePerFrame As Long
End Structure

#End If

#If ALLOW_UNTESTED_INTERFACES Then

<ComImport, System.Security.SuppressUnmanagedCodeSecurity, Guid("36d39eb0-dd75-11ce-bf0e-00aa0055595a"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)> _
Public Interface IDirectDrawVideo
	<PreserveSig> _
	Function GetSwitches(ByRef pSwitches As Integer) As Integer

	<PreserveSig> _
	Function SetSwitches(Switches As Integer) As Integer

	<PreserveSig> _
	Function GetCaps(ByRef pCaps As IntPtr) As Integer ' DDCAPS

	<PreserveSig> _
	Function GetEmulatedCaps(ByRef pCaps As IntPtr) As Integer ' DDCAPS

	<PreserveSig> _
	Function GetSurfaceDesc(ByRef pSurfaceDesc As IntPtr) As Integer ' DDSURFACEDESC

	<PreserveSig> _
	Function GetFourCCCodes(ByRef pCount As Integer, ByRef pCodes As Integer) As Integer

	<PreserveSig> _
	Function SetDirectDraw(pDirectDraw As IntPtr) As Integer ' LPDIRECTDRAW

	<PreserveSig> _
	Function GetDirectDraw(ByRef ppDirectDraw As IntPtr) As Integer ' LPDIRECTDRAW

	<PreserveSig> _
	Function GetSurfaceType(ByRef pSurfaceType As DirectDrawSwitches) As Integer

	<PreserveSig> _
	Function SetDefault() As Integer

    <PreserveSig()> _
    Function UseScanLine(ByVal _UseScanLine As Integer) As Integer

    <PreserveSig()> _
    Function CanUseScanLine(ByRef _UseScanLine As Integer) As Integer

    <PreserveSig()> _
    Function UseOverlayStretch(ByVal _UseOverlayStretch As Integer) As Integer

    <PreserveSig()> _
    Function CanUseOverlayStretch(ByRef _UseOverlayStretch As Integer) As Integer

    <PreserveSig()> _
    Function UseWhenFullScreen(ByVal _UseWhenFullScreen As Integer) As Integer

    <PreserveSig()> _
    Function WillUseFullScreen(ByRef _UseWhenFullScreen As Integer) As Integer
End Interface

<ComImport, System.Security.SuppressUnmanagedCodeSecurity, Guid("dd1d7110-7836-11cf-bf47-00aa0055595a"), Obsolete("This interface has been deprecated.", False), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)> _
Public Interface IFullScreenVideo
	<PreserveSig> _
	Function CountModes(ByRef pModes As Integer) As Integer

	<PreserveSig> _
	Function GetModeInfo(Mode As Integer, ByRef pWidth As Integer, ByRef pHeight As Integer, ByRef pDepth As Integer) As Integer

	<PreserveSig> _
	Function GetCurrentMode(ByRef pMode As Integer) As Integer

	<PreserveSig> _
	Function IsModeAvailable(Mode As Integer) As Integer

	<PreserveSig> _
	Function IsModeEnabled(Mode As Integer) As Integer

	<PreserveSig> _
	Function SetEnabled(Mode As Integer, bEnabled As Integer) As Integer

	<PreserveSig> _
	Function GetClipFactor(ByRef pClipFactor As Integer) As Integer

	<PreserveSig> _
	Function SetClipFactor(ClipFactor As Integer) As Integer

	<PreserveSig> _
	Function SetMessageDrain(hwnd As IntPtr) As Integer

	<PreserveSig> _
	Function GetMessageDrain(ByRef hwnd As IntPtr) As Integer

	<PreserveSig> _
	Function SetMonitor(Monitor As Integer) As Integer

	<PreserveSig> _
	Function GetMonitor(ByRef Monitor As Integer) As Integer

	<PreserveSig> _
	Function HideOnDeactivate(Hide As Integer) As Integer

	<PreserveSig> _
	Function IsHideOnDeactivate() As Integer

	<PreserveSig> _
	Function SetCaption(<MarshalAs(UnmanagedType.BStr)> strCaption As String) As Integer

	<PreserveSig> _
	Function GetCaption(<MarshalAs(UnmanagedType.BStr)> ByRef pstrCaption As String) As Integer

	<PreserveSig> _
	Function SetDefault() As Integer

End Interface


' -------------------------------------------------------------------------------
'  This interface has been deprecated
' -------------------------------------------------------------------------------
'<ComImport, System.Security.SuppressUnmanagedCodeSecurity, Guid("53479470-f1dd-11cf-bc42-00aa00ac74f6"), Obsolete("This interface has been deprecated.", False), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)> _
'Public Interface IFullScreenVideoEx
'	Inherits IFullScreenVideo

'    <PreserveSig()> _
'    Shadows Function CountModes(ByRef pModes As Integer) As Integer

'    <PreserveSig()> _
'    Shadows Function GetModeInfo(ByVal Mode As Integer, ByRef pWidth As Integer, ByRef pHeight As Integer, ByRef pDepth As Integer) As Integer

'    <PreserveSig()> _
'    Shadows Function GetCurrentMode(ByRef pMode As Integer) As Integer

'    <PreserveSig()> _
'    Shadows Function IsModeAvailable(ByVal Mode As Integer) As Integer

'    <PreserveSig()> _
'    Shadows Function IsModeEnabled(ByVal Mode As Integer) As Integer

'    <PreserveSig()> _
'    Shadows Function SetEnabled(ByVal Mode As Integer, ByVal bEnabled As Integer) As Integer

'    <PreserveSig()> _
'    Shadows Function GetClipFactor(ByRef pClipFactor As Integer) As Integer

'    <PreserveSig()> _
'    Shadows Function SetClipFactor(ByVal ClipFactor As Integer) As Integer

'    <PreserveSig()> _
'    Shadows Function SetMessageDrain(ByVal hwnd As IntPtr) As Integer

'    <PreserveSig()> _
'    Shadows Function GetMessageDrain(ByRef hwnd As IntPtr) As Integer

'    <PreserveSig()> _
'    Shadows Function SetMonitor(ByVal Monitor As Integer) As Integer

'    <PreserveSig()> _
'    Shadows Function GetMonitor(ByRef Monitor As Integer) As Integer

'    <PreserveSig()> _
'    Shadows Function HideOnDeactivate(ByVal Hide As Integer) As Integer

'    <PreserveSig()> _
'    Shadows Function IsHideOnDeactivate() As Integer

'    <PreserveSig()> _
'    Shadows Function SetCaption(<MarshalAs(UnmanagedType.BStr)> ByVal strCaption As String) As Integer

'    <PreserveSig()> _
'    Shadows Function GetCaption(<MarshalAs(UnmanagedType.BStr)> ByRef pstrCaption As String) As Integer

'    <PreserveSig()> _
'    Shadows Function SetDefault() As Integer

'	<PreserveSig> _
'	Function SetAcceleratorTable(hwnd As IntPtr, hAccel As IntPtr) As Integer ' HACCEL

'	<PreserveSig> _
'	Function GetAcceleratorTable(ByRef phwnd As IntPtr, ByRef phAccel As IntPtr) As Integer ' HACCEL

'	<PreserveSig> _
'	Function KeepPixelAspectRatio(KeepAspect As Integer) As Integer

'	<PreserveSig> _
'	Function IsKeepPixelAspectRatio(ByRef pKeepAspect As Integer) As Integer
'End Interface

<ComImport, System.Security.SuppressUnmanagedCodeSecurity, Guid("61ded640-e912-11ce-a099-00aa00479a58"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)> _
Public Interface IBaseVideoMixer
	<PreserveSig> _
	Function SetLeadPin(iPin As Integer) As Integer

	<PreserveSig> _
	Function GetLeadPin(ByRef piPin As Integer) As Integer

	<PreserveSig> _
	Function GetInputPinCount(ByRef piPinCount As Integer) As Integer

	<PreserveSig> _
	Function IsUsingClock(ByRef pbValue As Integer) As Integer

	<PreserveSig> _
	Function SetUsingClock(bValue As Integer) As Integer

	<PreserveSig> _
	Function GetClockPeriod(ByRef pbValue As Integer) As Integer

	<PreserveSig> _
	Function SetClockPeriod(bValue As Integer) As Integer
End Interface

#End If

<ComImport, System.Security.SuppressUnmanagedCodeSecurity, Guid("1bd0ecb0-f8e2-11ce-aac6-0020af0b99a3"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)> _
Public Interface IQualProp
	<PreserveSig> _
	Function get_FramesDroppedInRenderer(ByRef pcFrames As Integer) As Integer

	<PreserveSig> _
	Function get_FramesDrawn(ByRef pcFramesDrawn As Integer) As Integer

	<PreserveSig> _
	Function get_AvgFrameRate(ByRef piAvgFrameRate As Integer) As Integer

	<PreserveSig> _
	Function get_Jitter(ByRef iJitter As Integer) As Integer

	<PreserveSig> _
	Function get_AvgSyncOffset(ByRef piAvg As Integer) As Integer

	<PreserveSig> _
	Function get_DevSyncOffset(ByRef piDev As Integer) As Integer

End Interface

