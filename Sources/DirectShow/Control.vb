
Imports System.Runtime.InteropServices

<Flags> _
Public Enum NotifyFlags
	None
	NoNotify
End Enum

Public Enum OABool
	[False] = 0
    [True] = -1  ' bools in .NET use 1, not -1
End Enum

<Flags()> _
Public Enum WindowStyle
    Overlapped = &H0
    Popup = &H80000000
    Child = &H40000000
    Minimize = &H20000000
    Visible = &H10000000
    Disabled = &H8000000
    ClipSiblings = &H4000000
    ClipChildren = &H2000000
    Maximize = &H1000000
    Caption = &HC00000
    Border = &H800000
    DlgFrame = &H400000
    VScroll = &H200000
    HScroll = &H100000
    SysMenu = &H80000
    ThickFrame = &H40000
    Group = &H20000
    TabStop = &H10000
    MinimizeBox = &H20000
    MaximizeBox = &H10000
End Enum

<Flags> _
Public Enum WindowStyleEx
	DlgModalFrame = &H1
	NoParentNotify = &H4
	Topmost = &H8
	AcceptFiles = &H10
	Transparent = &H20
	MDIChild = &H40
	ToolWindow = &H80
	WindowEdge = &H100
	ClientEdge = &H200
	ContextHelp = &H400
	Right = &H1000
	Left = &H0
	RTLReading = &H2000
	LTRReading = &H0
	LeftScrollBar = &H4000
	RightScrollBar = &H0
	ControlParent = &H10000
	StaticEdge = &H20000
	APPWindow = &H40000
	Layered = &H80000
	NoInheritLayout = &H100000
	LayoutRTL = &H400000
	Composited = &H2000000
	NoActivate = &H8000000
End Enum

Public Enum WindowState
	Hide = 0
	Normal
	ShowMinimized
	ShowMaximized
	ShowNoActivate
	Show
	Minimize
	ShowMinNoActive
	ShowNA
	Restore
	ShowDefault
	ForceMinimize
End Enum

<Flags> _
Public Enum DispatchFlags As Short
	None = &H0
	Method = &H1
	PropertyGet = &H2
	PropertyPut = &H4
	PropertyPutRef = &H8
End Enum



#If ALLOW_UNTESTED_INTERFACES Then

<ComImport, System.Security.SuppressUnmanagedCodeSecurity, Guid("56a868b9-0ad4-11ce-b03a-0020af0ba770"), InterfaceType(ComInterfaceType.InterfaceIsDual)> _
Public Interface IAMCollection
	<PreserveSig> _
	Function get_Count(<Out> ByRef plCount As Integer) As Integer

	<PreserveSig> _
	Function Item(<[In]> lItem As Integer, <Out, MarshalAs(UnmanagedType.IUnknown)> ByRef ppUnk As Object) As Integer

	<PreserveSig> _
	Function get__NewEnum(<Out, MarshalAs(UnmanagedType.IUnknown)> ByRef ppUnk As Object) As Integer
End Interface

<ComImport, System.Security.SuppressUnmanagedCodeSecurity, Guid("56a868ba-0ad4-11ce-b03a-0020af0ba770"), InterfaceType(ComInterfaceType.InterfaceIsDual)> _
Public Interface IFilterInfo
	<PreserveSig> _
	Function FindPin(<[In], MarshalAs(UnmanagedType.BStr)> strPinID As String, <Out, MarshalAs(UnmanagedType.IDispatch)> ByRef ppUnk As Object) As Integer

	<PreserveSig> _
	Function get_Name(<Out, MarshalAs(UnmanagedType.BStr)> ByRef strName As String) As Integer

	<PreserveSig> _
	Function get_VendorInfo(<Out, MarshalAs(UnmanagedType.BStr)> strVendorInfo As String) As Integer

	<PreserveSig> _
	Function get_Filter(<Out, MarshalAs(UnmanagedType.IUnknown)> ByRef ppUnk As Object) As Integer

	<PreserveSig> _
	Function get_Pins(<Out, MarshalAs(UnmanagedType.IDispatch)> ByRef ppUnk As Object) As Integer

	<PreserveSig> _
	Function get_IsFileSource(<Out> ByRef pbIsSource As Integer) As Integer

	<PreserveSig> _
	Function get_Filename(<Out, MarshalAs(UnmanagedType.BStr)> ByRef pstrFilename As String) As Integer

	<PreserveSig> _
	Function put_Filename(<[In], MarshalAs(UnmanagedType.BStr)> strFilename As String) As Integer
End Interface

<ComImport, System.Security.SuppressUnmanagedCodeSecurity, Guid("56a868bb-0ad4-11ce-b03a-0020af0ba770"), InterfaceType(ComInterfaceType.InterfaceIsDual)> _
Public Interface IRegFilterInfo
	<PreserveSig> _
	Function get_Name(<Out, MarshalAs(UnmanagedType.BStr)> ByRef strName As String) As Integer

	<PreserveSig> _
	Function Filter(<Out, MarshalAs(UnmanagedType.IDispatch)> ByRef ppUnk As Object) As Integer
End Interface

<ComImport, System.Security.SuppressUnmanagedCodeSecurity, Guid("56a868bc-0ad4-11ce-b03a-0020af0ba770"), InterfaceType(ComInterfaceType.InterfaceIsDual)> _
Public Interface IMediaTypeInfo
	<PreserveSig> _
	Function get_Type(<Out, MarshalAs(UnmanagedType.BStr)> ByRef strType As String) As Integer

	<PreserveSig> _
	Function get_Subtype(<Out, MarshalAs(UnmanagedType.BStr)> ByRef strType As String) As Integer
End Interface

<ComImport, System.Security.SuppressUnmanagedCodeSecurity, Guid("56a868bd-0ad4-11ce-b03a-0020af0ba770"), InterfaceType(ComInterfaceType.InterfaceIsDual)> _
Public Interface IPinInfo
	<PreserveSig> _
	Function get_Pin(<Out, MarshalAs(UnmanagedType.IUnknown)> ByRef ppUnk As Object) As Integer

	<PreserveSig> _
	Function get_ConnectedTo(<Out, MarshalAs(UnmanagedType.IDispatch)> ByRef ppUnk As Object) As Integer

	<PreserveSig> _
	Function get_ConnectionMediaType(<Out, MarshalAs(UnmanagedType.IUnknown)> ByRef ppUnk As Object) As Integer

	<PreserveSig> _
	Function get_FilterInfo(<Out, MarshalAs(UnmanagedType.IUnknown)> ByRef ppUnk As Object) As Integer

	<PreserveSig> _
	Function get_Name(<Out, MarshalAs(UnmanagedType.BStr)> ByRef ppUnk As String) As Integer

	<PreserveSig> _
	Function get_Direction(<Out> ppDirection As Integer) As Integer

	<PreserveSig> _
	Function get_PinID(<Out, MarshalAs(UnmanagedType.BStr)> ByRef strPinID As String) As Integer

	<PreserveSig> _
	Function get_MediaTypes(<Out, MarshalAs(UnmanagedType.IUnknown)> ByRef ppUnk As Object) As Integer

	<PreserveSig> _
	Function Connect(<[In], MarshalAs(UnmanagedType.IUnknown)> pPin As Object) As Integer

	<PreserveSig> _
	Function ConnectDirect(<[In], MarshalAs(UnmanagedType.IUnknown)> pPin As Object) As Integer

	<PreserveSig> _
	Function ConnectWithType(<[In], MarshalAs(UnmanagedType.IUnknown)> pPin As Object, <[In], MarshalAs(UnmanagedType.IUnknown)> pMediaType As Object) As Integer

	<PreserveSig> _
	Function Disconnect() As Integer

	<PreserveSig> _
	Function Render() As Integer
End Interface

#End If

<ComImport, System.Security.SuppressUnmanagedCodeSecurity, Guid("bc9bcf80-dcd2-11d2-abf6-00a0c905f375"), InterfaceType(ComInterfaceType.InterfaceIsDual)> _
Public Interface IAMStats
	<PreserveSig> _
	Function Reset() As Integer

	<PreserveSig> _
	Function get_Count(<Out> ByRef plCount As Integer) As Integer

	<PreserveSig> _
	Function GetValueByIndex(<[In]> lIndex As Integer, <Out, MarshalAs(UnmanagedType.BStr)> ByRef szName As String, <Out> ByRef lCount As Integer, <Out> ByRef dLast As Double, <Out> ByRef dAverage As Double, <Out> ByRef dStdDev As Double, _
		<Out> ByRef dMin As Double, <Out> ByRef dMax As Double) As Integer

	<PreserveSig> _
	Function GetValueByName(<[In], MarshalAs(UnmanagedType.BStr)> szName As String, <Out> ByRef lIndex As Integer, <Out> ByRef lCount As Integer, <Out> ByRef dLast As Double, <Out> ByRef dAverage As Double, <Out> ByRef dStdDev As Double, _
		<Out> ByRef dMin As Double, <Out> ByRef dMax As Double) As Integer

	<PreserveSig> _
	Function GetIndex(<[In], MarshalAs(UnmanagedType.BStr)> szName As String, <[In], MarshalAs(UnmanagedType.Bool)> lCreate As Boolean, <Out> ByRef plIndex As Integer) As Integer

	<PreserveSig> _
	Function AddValue(<[In]> lIndex As Integer, <[In]> dValue As Double) As Integer
End Interface

<ComImport, System.Security.SuppressUnmanagedCodeSecurity, Guid("56a868b4-0ad4-11ce-b03a-0020af0ba770"), InterfaceType(ComInterfaceType.InterfaceIsDual)> _
Public Interface IVideoWindow
	<PreserveSig> _
	Function put_Caption(<[In], MarshalAs(UnmanagedType.BStr)> caption As String) As Integer

	<PreserveSig> _
	Function get_Caption(<Out, MarshalAs(UnmanagedType.BStr)> ByRef caption As String) As Integer

	<PreserveSig> _
	Function put_WindowStyle(<[In]> windowStyle As WindowStyle) As Integer

	<PreserveSig> _
	Function get_WindowStyle(<Out> ByRef windowStyle As WindowStyle) As Integer

	<PreserveSig> _
	Function put_WindowStyleEx(<[In]> windowStyleEx As WindowStyleEx) As Integer

	<PreserveSig> _
	Function get_WindowStyleEx(<Out> ByRef windowStyleEx As WindowStyleEx) As Integer

	<PreserveSig> _
	Function put_AutoShow(<[In]> autoShow As OABool) As Integer

	<PreserveSig> _
	Function get_AutoShow(<Out> ByRef autoShow As OABool) As Integer

	<PreserveSig> _
	Function put_WindowState(<[In]> windowState As WindowState) As Integer

	<PreserveSig> _
	Function get_WindowState(<Out> ByRef windowState As WindowState) As Integer

	<PreserveSig> _
	Function put_BackgroundPalette(<[In]> backgroundPalette As OABool) As Integer

	<PreserveSig> _
	Function get_BackgroundPalette(<Out> ByRef backgroundPalette As OABool) As Integer

	<PreserveSig> _
	Function put_Visible(<[In]> visible As OABool) As Integer

	<PreserveSig> _
	Function get_Visible(<Out> ByRef visible As OABool) As Integer

	<PreserveSig> _
	Function put_Left(<[In]> left As Integer) As Integer

	<PreserveSig> _
	Function get_Left(<Out> ByRef left As Integer) As Integer

	<PreserveSig> _
	Function put_Width(<[In]> width As Integer) As Integer

	<PreserveSig> _
	Function get_Width(<Out> ByRef width As Integer) As Integer

	<PreserveSig> _
	Function put_Top(<[In]> top As Integer) As Integer

	<PreserveSig> _
	Function get_Top(<Out> ByRef top As Integer) As Integer

	<PreserveSig> _
	Function put_Height(<[In]> height As Integer) As Integer

	<PreserveSig> _
	Function get_Height(<Out> ByRef height As Integer) As Integer

	<PreserveSig> _
	Function put_Owner(<[In]> owner As IntPtr) As Integer

	<PreserveSig> _
	Function get_Owner(<Out> ByRef owner As IntPtr) As Integer

	<PreserveSig> _
	Function put_MessageDrain(<[In]> drain As IntPtr) As Integer

	<PreserveSig> _
	Function get_MessageDrain(<Out> ByRef drain As IntPtr) As Integer

	' Use ColorTranslator to break out RGB
	<PreserveSig> _
	Function get_BorderColor(<Out> ByRef color As Integer) As Integer

	' Use ColorTranslator to break out RGB
	<PreserveSig> _
	Function put_BorderColor(<[In]> color As Integer) As Integer

	<PreserveSig> _
	Function get_FullScreenMode(<Out> ByRef fullScreenMode As OABool) As Integer

	<PreserveSig> _
	Function put_FullScreenMode(<[In]> fullScreenMode As OABool) As Integer

	<PreserveSig> _
	Function SetWindowForeground(<[In]> focus As OABool) As Integer

	' HWND *
	' WPARAM
    ' LPARAM
	<PreserveSig> _
	Function NotifyOwnerMessage(<[In]> hwnd As IntPtr, <[In]> msg As Integer, <[In]> wParam As IntPtr, <[In]> lParam As IntPtr) As Integer

	<PreserveSig> _
	Function SetWindowPosition(<[In]> left As Integer, <[In]> top As Integer, <[In]> width As Integer, <[In]> height As Integer) As Integer

	<PreserveSig> _
	Function GetWindowPosition(<Out> ByRef left As Integer, <Out> ByRef top As Integer, <Out> ByRef width As Integer, <Out> ByRef height As Integer) As Integer

	<PreserveSig> _
	Function GetMinIdealImageSize(<Out> ByRef width As Integer, <Out> ByRef height As Integer) As Integer

	<PreserveSig> _
	Function GetMaxIdealImageSize(<Out> ByRef width As Integer, <Out> ByRef height As Integer) As Integer

	<PreserveSig> _
	Function GetRestorePosition(<Out> ByRef left As Integer, <Out> ByRef top As Integer, <Out> ByRef width As Integer, <Out> ByRef height As Integer) As Integer

	<PreserveSig> _
	Function HideCursor(<[In]> hideCursor__1 As OABool) As Integer

	<PreserveSig> _
	Function IsCursorHidden(<Out> ByRef hideCursor As OABool) As Integer
End Interface

<ComImport, System.Security.SuppressUnmanagedCodeSecurity, Guid("56a868b3-0ad4-11ce-b03a-0020af0ba770"), InterfaceType(ComInterfaceType.InterfaceIsDual)> _
Public Interface IBasicAudio
	<PreserveSig> _
	Function put_Volume(<[In]> lVolume As Integer) As Integer

	<PreserveSig> _
	Function get_Volume(<Out> ByRef plVolume As Integer) As Integer

	<PreserveSig> _
	Function put_Balance(<[In]> lBalance As Integer) As Integer

	<PreserveSig> _
	Function get_Balance(<Out> ByRef plBalance As Integer) As Integer
End Interface

<ComImport, System.Security.SuppressUnmanagedCodeSecurity, Guid("56a868b5-0ad4-11ce-b03a-0020af0ba770"), InterfaceType(ComInterfaceType.InterfaceIsDual)> _
Public Interface IBasicVideo
	<PreserveSig> _
	Function get_AvgTimePerFrame(<Out> ByRef pAvgTimePerFrame As Double) As Integer

	<PreserveSig> _
	Function get_BitRate(<Out> ByRef pBitRate As Integer) As Integer

	<PreserveSig> _
	Function get_BitErrorRate(<Out> ByRef pBitRate As Integer) As Integer

	<PreserveSig> _
	Function get_VideoWidth(<Out> ByRef pVideoWidth As Integer) As Integer

	<PreserveSig> _
	Function get_VideoHeight(<Out> ByRef pVideoHeight As Integer) As Integer

	<PreserveSig> _
	Function put_SourceLeft(<[In]> SourceLeft As Integer) As Integer

	<PreserveSig> _
	Function get_SourceLeft(<Out> ByRef pSourceLeft As Integer) As Integer

	<PreserveSig> _
	Function put_SourceWidth(<[In]> SourceWidth As Integer) As Integer

	<PreserveSig> _
	Function get_SourceWidth(<Out> ByRef pSourceWidth As Integer) As Integer

	<PreserveSig> _
	Function put_SourceTop(<[In]> SourceTop As Integer) As Integer

	<PreserveSig> _
	Function get_SourceTop(<Out> ByRef pSourceTop As Integer) As Integer

	<PreserveSig> _
	Function put_SourceHeight(<[In]> SourceHeight As Integer) As Integer

	<PreserveSig> _
	Function get_SourceHeight(<Out> ByRef pSourceHeight As Integer) As Integer

	<PreserveSig> _
	Function put_DestinationLeft(<[In]> DestinationLeft As Integer) As Integer

	<PreserveSig> _
	Function get_DestinationLeft(<Out> ByRef pDestinationLeft As Integer) As Integer

	<PreserveSig> _
	Function put_DestinationWidth(<[In]> DestinationWidth As Integer) As Integer

	<PreserveSig> _
	Function get_DestinationWidth(<Out> ByRef pDestinationWidth As Integer) As Integer

	<PreserveSig> _
	Function put_DestinationTop(<[In]> DestinationTop As Integer) As Integer

	<PreserveSig> _
	Function get_DestinationTop(<Out> ByRef pDestinationTop As Integer) As Integer

	<PreserveSig> _
	Function put_DestinationHeight(<[In]> DestinationHeight As Integer) As Integer

	<PreserveSig> _
	Function get_DestinationHeight(<Out> ByRef pDestinationHeight As Integer) As Integer

	<PreserveSig> _
	Function SetSourcePosition(<[In]> left As Integer, <[In]> top As Integer, <[In]> width As Integer, <[In]> height As Integer) As Integer

	<PreserveSig> _
	Function GetSourcePosition(<Out> ByRef left As Integer, <Out> ByRef top As Integer, <Out> ByRef width As Integer, <Out> ByRef height As Integer) As Integer

	<PreserveSig> _
	Function SetDefaultSourcePosition() As Integer

	<PreserveSig> _
	Function SetDestinationPosition(<[In]> left As Integer, <[In]> top As Integer, <[In]> width As Integer, <[In]> height As Integer) As Integer

	<PreserveSig> _
	Function GetDestinationPosition(<Out> ByRef left As Integer, <Out> ByRef top As Integer, <Out> ByRef width As Integer, <Out> ByRef height As Integer) As Integer

	<PreserveSig> _
	Function SetDefaultDestinationPosition() As Integer

	<PreserveSig> _
	Function GetVideoSize(<Out> ByRef pWidth As Integer, <Out> ByRef pHeight As Integer) As Integer

	<PreserveSig> _
	Function GetVideoPaletteEntries(<[In]> StartIndex As Integer, <[In]> Entries As Integer, <Out> ByRef pRetrieved As Integer, <Out> ByRef pPalette As Integer()) As Integer

		' int *
	<PreserveSig> _
	Function GetCurrentImage(<[In], Out> ByRef pBufferSize As Integer, <Out> pDIBImage As IntPtr) As Integer

	<PreserveSig> _
	Function IsUsingDefaultSource() As Integer

	<PreserveSig> _
	Function IsUsingDefaultDestination() As Integer
End Interface

<ComImport, System.Security.SuppressUnmanagedCodeSecurity, Guid("329bb360-f6ea-11d1-9038-00a0c9697298"), InterfaceType(ComInterfaceType.InterfaceIsDual)> _
Public Interface IBasicVideo2
	Inherits IBasicVideo

    <PreserveSig()> _
    Shadows Function get_AvgTimePerFrame(<Out()> ByRef pAvgTimePerFrame As Double) As Integer

    <PreserveSig()> _
    Shadows Function get_BitRate(<Out()> ByRef pBitRate As Integer) As Integer

    <PreserveSig()> _
    Shadows Function get_BitErrorRate(<Out()> ByRef pBitRate As Integer) As Integer

    <PreserveSig()> _
    Shadows Function get_VideoWidth(<Out()> ByRef pVideoWidth As Integer) As Integer

    <PreserveSig()> _
    Shadows Function get_VideoHeight(<Out()> ByRef pVideoHeight As Integer) As Integer

    <PreserveSig()> _
    Shadows Function put_SourceLeft(<[In]()> ByVal SourceLeft As Integer) As Integer

    <PreserveSig()> _
    Shadows Function get_SourceLeft(<Out()> ByRef pSourceLeft As Integer) As Integer

    <PreserveSig()> _
    Shadows Function put_SourceWidth(<[In]()> ByVal SourceWidth As Integer) As Integer

    <PreserveSig()> _
    Shadows Function get_SourceWidth(<Out()> ByRef pSourceWidth As Integer) As Integer

    <PreserveSig()> _
    Shadows Function put_SourceTop(<[In]()> ByVal SourceTop As Integer) As Integer

    <PreserveSig()> _
    Shadows Function get_SourceTop(<Out()> ByRef pSourceTop As Integer) As Integer

    <PreserveSig()> _
    Shadows Function put_SourceHeight(<[In]()> ByVal SourceHeight As Integer) As Integer

    <PreserveSig()> _
    Shadows Function get_SourceHeight(<Out()> ByRef pSourceHeight As Integer) As Integer

    <PreserveSig()> _
    Shadows Function put_DestinationLeft(<[In]()> ByVal DestinationLeft As Integer) As Integer

    <PreserveSig()> _
    Shadows Function get_DestinationLeft(<Out()> ByRef pDestinationLeft As Integer) As Integer

    <PreserveSig()> _
    Shadows Function put_DestinationWidth(<[In]()> ByVal DestinationWidth As Integer) As Integer

    <PreserveSig()> _
    Shadows Function get_DestinationWidth(<Out()> ByRef pDestinationWidth As Integer) As Integer

    <PreserveSig()> _
    Shadows Function put_DestinationTop(<[In]()> ByVal DestinationTop As Integer) As Integer

    <PreserveSig()> _
    Shadows Function get_DestinationTop(<Out()> ByRef pDestinationTop As Integer) As Integer

    <PreserveSig()> _
    Shadows Function put_DestinationHeight(<[In]()> ByVal DestinationHeight As Integer) As Integer

    <PreserveSig()> _
    Shadows Function get_DestinationHeight(<Out()> ByRef pDestinationHeight As Integer) As Integer

    <PreserveSig()> _
    Shadows Function SetSourcePosition(<[In]()> ByVal left As Integer, <[In]()> ByVal top As Integer, <[In]()> ByVal width As Integer, <[In]()> ByVal height As Integer) As Integer

    <PreserveSig()> _
    Shadows Function GetSourcePosition(<Out()> ByRef left As Integer, <Out()> ByRef top As Integer, <Out()> ByRef width As Integer, <Out()> ByRef height As Integer) As Integer

    <PreserveSig()> _
    Shadows Function SetDefaultSourcePosition() As Integer

    <PreserveSig()> _
    Shadows Function SetDestinationPosition(<[In]()> ByVal left As Integer, <[In]()> ByVal top As Integer, <[In]()> ByVal width As Integer, <[In]()> ByVal height As Integer) As Integer

    <PreserveSig()> _
    Shadows Function GetDestinationPosition(<Out()> ByRef left As Integer, <Out()> ByRef top As Integer, <Out()> ByRef width As Integer, <Out()> ByRef height As Integer) As Integer

    <PreserveSig()> _
    Shadows Function SetDefaultDestinationPosition() As Integer

    <PreserveSig()> _
    Shadows Function GetVideoSize(<Out()> ByRef pWidth As Integer, <Out()> ByRef pHeight As Integer) As Integer

    <PreserveSig()> _
    Shadows Function GetVideoPaletteEntries(<[In]()> ByVal StartIndex As Integer, <[In]()> ByVal Entries As Integer, <Out()> ByRef pRetrieved As Integer, <Out()> ByRef pPalette As Integer()) As Integer

    ' int *
    <PreserveSig()> _
    Shadows Function GetCurrentImage(<[In](), Out()> ByRef pBufferSize As Integer, <Out()> ByVal pDIBImage As IntPtr) As Integer

    <PreserveSig()> _
    Shadows Function IsUsingDefaultSource() As Integer

    <PreserveSig()> _
    Shadows Function IsUsingDefaultDestination() As Integer

	<PreserveSig> _
	Function GetPreferredAspectRatio(<Out> ByRef plAspectX As Integer, <Out> ByRef plAspectY As Integer) As Integer
End Interface

<ComImport, System.Security.SuppressUnmanagedCodeSecurity, Guid("56a868b6-0ad4-11ce-b03a-0020af0ba770"), InterfaceType(ComInterfaceType.InterfaceIsDual)> _
Public Interface IMediaEvent
	<PreserveSig> _
	Function GetEventHandle(<Out> ByRef hEvent As IntPtr) As Integer
	' HEVENT
	<PreserveSig> _
	Function GetEvent(<Out> ByRef lEventCode As EventCode, <Out> ByRef lParam1 As IntPtr, <Out> ByRef lParam2 As IntPtr, <[In]> msTimeout As Integer) As Integer

	<PreserveSig> _
	Function WaitForCompletion(<[In]> msTimeout As Integer, <Out> ByRef pEvCode As EventCode) As Integer

	<PreserveSig> _
	Function CancelDefaultHandling(<[In]> lEvCode As EventCode) As Integer

	<PreserveSig> _
	Function RestoreDefaultHandling(<[In]> lEvCode As EventCode) As Integer

	<PreserveSig> _
	Function FreeEventParams(<[In]> lEvCode As EventCode, <[In]> lParam1 As IntPtr, <[In]> lParam2 As IntPtr) As Integer
End Interface

<ComImport, System.Security.SuppressUnmanagedCodeSecurity, Guid("56a868c0-0ad4-11ce-b03a-0020af0ba770"), InterfaceType(ComInterfaceType.InterfaceIsDual)> _
Public Interface IMediaEventEx
	Inherits IMediaEvent

    ' HEVENT
    <PreserveSig()> _
    Shadows Function GetEventHandle(<Out()> ByRef hEvent As IntPtr) As Integer

    <PreserveSig()> _
    Shadows Function GetEvent(<Out()> ByRef lEventCode As EventCode, <Out()> ByRef lParam1 As IntPtr, <Out()> ByRef lParam2 As IntPtr, <[In]()> ByVal msTimeout As Integer) As Integer

    <PreserveSig()> _
    Shadows Function WaitForCompletion(<[In]()> ByVal msTimeout As Integer, <Out()> ByRef pEvCode As EventCode) As Integer

    <PreserveSig()> _
    Shadows Function CancelDefaultHandling(<[In]()> ByVal lEvCode As EventCode) As Integer

    <PreserveSig()> _
    Shadows Function RestoreDefaultHandling(<[In]()> ByVal lEvCode As EventCode) As Integer

    <PreserveSig()> _
    Shadows Function FreeEventParams(<[In]()> ByVal lEvCode As EventCode, <[In]()> ByVal lParam1 As IntPtr, <[In]()> ByVal lParam2 As IntPtr) As Integer

	' HWND *
    ' PVOID
	<PreserveSig> _
	Function SetNotifyWindow(<[In]> hwnd As IntPtr, <[In]> lMsg As Integer, <[In]> lInstanceData As IntPtr) As Integer

	<PreserveSig> _
	Function SetNotifyFlags(<[In]> lNoNotifyFlags As NotifyFlags) As Integer

	<PreserveSig> _
	Function GetNotifyFlags(<Out> ByRef lplNoNotifyFlags As NotifyFlags) As Integer
End Interface

<ComImport, System.Security.SuppressUnmanagedCodeSecurity, Guid("56a868b2-0ad4-11ce-b03a-0020af0ba770"), InterfaceType(ComInterfaceType.InterfaceIsDual)> _
Public Interface IMediaPosition
	<PreserveSig> _
	Function get_Duration(<Out> ByRef pLength As Double) As Integer

	<PreserveSig> _
	Function put_CurrentPosition(<[In]> llTime As Double) As Integer

	<PreserveSig> _
	Function get_CurrentPosition(<Out> ByRef pllTime As Double) As Integer

	<PreserveSig> _
	Function get_StopTime(<Out> ByRef pllTime As Double) As Integer

	<PreserveSig> _
	Function put_StopTime(<[In]> llTime As Double) As Integer

	<PreserveSig> _
	Function get_PrerollTime(<Out> ByRef pllTime As Double) As Integer

	<PreserveSig> _
	Function put_PrerollTime(<[In]> llTime As Double) As Integer

	<PreserveSig> _
	Function put_Rate(<[In]> dRate As Double) As Integer

	<PreserveSig> _
	Function get_Rate(<Out> ByRef pdRate As Double) As Integer

	<PreserveSig> _
	Function CanSeekForward(<Out> ByRef pCanSeekForward As OABool) As Integer

	<PreserveSig> _
	Function CanSeekBackward(<Out> ByRef pCanSeekBackward As OABool) As Integer
End Interface

<ComImport, System.Security.SuppressUnmanagedCodeSecurity, Guid("56a868b1-0ad4-11ce-b03a-0020af0ba770"), InterfaceType(ComInterfaceType.InterfaceIsDual)> _
Public Interface IMediaControl
	<PreserveSig> _
	Function Run() As Integer

	<PreserveSig> _
	Function Pause() As Integer

	<PreserveSig> _
	Function [Stop]() As Integer

	<PreserveSig> _
	Function GetState(<[In]> msTimeout As Integer, <Out> ByRef pfs As FilterState) As Integer

	<PreserveSig> _
	Function RenderFile(<[In], MarshalAs(UnmanagedType.BStr)> strFilename As String) As Integer

	<PreserveSig, Obsolete("Automation interface, for pre-.NET VB.  Use IGraphBuilder::AddSourceFilter instead", False)> _
	Function AddSourceFilter(<[In], MarshalAs(UnmanagedType.BStr)> strFilename As String, <Out, MarshalAs(UnmanagedType.IDispatch)> ByRef ppUnk As Object) As Integer

	<PreserveSig, Obsolete("Automation interface, for pre-.NET VB.  Use IFilterGraph::EnumFilters instead", False)> _
	Function get_FilterCollection(<Out, MarshalAs(UnmanagedType.IDispatch)> ByRef ppUnk As Object) As Integer

	<PreserveSig, Obsolete("Automation interface, for pre-.NET VB.  Use IFilterMapper2::EnumMatchingFilters instead", False)> _
	Function get_RegFilterCollection(<Out, MarshalAs(UnmanagedType.IDispatch)> ByRef ppUnk As Object) As Integer

	<PreserveSig> _
	Function StopWhenReady() As Integer
End Interface

<ComImport, System.Security.SuppressUnmanagedCodeSecurity, Guid("56a868b7-0ad4-11ce-b03a-0020af0ba770"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)> _
Public Interface IQueueCommand
	<PreserveSig> _
	Function InvokeAtStreamTime(<Out> ByRef pCmd As IDeferredCommand, <[In]> time As Double, <[In], MarshalAs(UnmanagedType.LPStruct)> iid As Guid, <[In]> dispidMethod As Integer, <[In]> wFlags As DispatchFlags, <[In]> cArgs As Integer, _
		<[In]> pDispParams As Object(), <[In]> pvarResult As IntPtr, <Out> ByRef puArgErr As Short) As Integer

	Function InvokeAtPresentationTime(<Out> ByRef pCmd As IDeferredCommand, <[In]> time As Double, <[In], MarshalAs(UnmanagedType.LPStruct)> iid As Guid, <[In]> dispidMethod As Integer, <[In]> wFlags As DispatchFlags, <[In]> cArgs As Integer, _
		<[In]> pDispParams As Object(), <[In]> pvarResult As IntPtr, <Out> ByRef puArgErr As Short) As Integer
End Interface

<ComImport, System.Security.SuppressUnmanagedCodeSecurity, Guid("56a868b8-0ad4-11ce-b03a-0020af0ba770"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)> _
Public Interface IDeferredCommand
	<PreserveSig> _
	Function Cancel() As Integer

	<PreserveSig> _
	Function Confidence(<Out> ByRef pConfidence As Integer) As Integer

	<PreserveSig> _
	Function Postpone(<[In]> newtime As Double) As Integer

	<PreserveSig> _
	Function GetHResult(<Out> ByRef phrResult As Integer) As Integer
End Interface

