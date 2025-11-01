
Imports System.Drawing
Imports System.Runtime.InteropServices

<Flags> _
Public Enum VMR9PresentationFlags
	None = 0
	SyncPoint = &H1
	Preroll = &H2
	Discontinuity = &H4
	TimeValid = &H8
	SrcDstRectsValid = &H10
End Enum

<Flags> _
Public Enum VMR9SurfaceAllocationFlags
	None = 0
	ThreeDRenderTarget = &H1
	DXVATarget = &H2
	TextureSurface = &H4
	OffscreenSurface = &H8
	RGBDynamicSwitch = &H10
	UsageReserved = &He0
	UsageMask = &Hff
End Enum

<Flags> _
Public Enum VMR9ProcAmpControlFlags
	None = 0
	Brightness = &H1
	Contrast = &H2
	Hue = &H4
	Saturation = &H8
	Mask = &Hf
End Enum

<Flags> _
Public Enum VMR9MixerPrefs
	None = 0
    NoDecimation = &H1              ' No decimation - full size
    DecimateOutput = &H2            ' decimate output by 2 in x & y
    ARAdjustXorY = &H4              ' adjust the aspect ratio in x or y
    NonSquareMixing = &H8           ' assume AP can handle non-square mixing, avoids intermediate scales
	DecimateMask = &Hf

    BiLinearFiltering = &H10        ' use bi-linear filtering
    PointFiltering = &H20           ' use point filtering
	AnisotropicFiltering = &H40
	'
    PyramidalQuadFiltering = &H80   ' 4-sample tent
    GaussianQuadFiltering = &H100   ' 4-sample gaussian
    FilteringReserved = &HE00       ' bits reserved for future use.
    FilteringMask = &HFF0           ' OR of all above flags
	RenderTargetRGB = &H1000
    RenderTargetYUV = &H2000        ' Uses DXVA to perform mixing
    RenderTargetReserved = &HFC000  ' bits reserved for future use.
    RenderTargetMask = &HFF000      ' OR of all above flags
	DynamicSwitchToBOB = &H100000
	DynamicDecimateBy2 = &H200000

	DynamicReserved = &Hc00000
	DynamicMask = &Hf00000
End Enum

<Flags> _
Public Enum VMR9DeinterlaceTech
	Unknown = &H0
	BOBLineReplicate = &H1
	BOBVerticalStretch = &H2
	MedianFiltering = &H4
	EdgeFiltering = &H10
	FieldAdaptive = &H20
	PixelAdaptive = &H40
	MotionVectorSteered = &H80
End Enum

<Flags> _
Public Enum VMR9AlphaBitmapFlags
	None = 0
	Disable = &H1
	hDC = &H2
	EntireDDS = &H4
	SrcColorKey = &H8
	SrcRect = &H10
	FilterMode = &H20
End Enum

<Flags> _
Public Enum VMR9DeinterlacePrefs
	None = 0
	NextBest = &H1
	BOB = &H2
	Weave = &H4
	Mask = &H7
End Enum

<Flags> _
Public Enum VMR9RenderPrefs
	None = 0
    DoNotRenderBorder = &H1 ' app paints color keys
    Mask = &H1              ' OR of all above flags
End Enum

<Flags> _
Public Enum VMR9Mode
	None = 0
	Windowed = &H1
	Windowless = &H2
	Renderless = &H4
	Mask = &H7
End Enum

Public Enum VMR9AspectRatioMode
	None
	LetterBox
End Enum

Public Enum VMR9SampleFormat
	None = 0
	Reserved = 1
	ProgressiveFrame = 2
	FieldInterleavedEvenFirst = 3
	FieldInterleavedOddFirst = 4
	FieldSingleEven = 5
	FieldSingleOdd = 6
End Enum

<StructLayout(LayoutKind.Sequential)> _
Public Structure VMR9PresentationInfo
	Public dwFlags As VMR9PresentationFlags
    Public lpSurf As IntPtr 'IDirect3DSurface9
	Public rtStart As Long
	Public rtEnd As Long
	Public szAspectRatio As Size
	Public rcSrc As DsRect
	Public rcDst As DsRect
	Public dwReserved1 As Integer
	Public dwReserved2 As Integer
End Structure

<StructLayout(LayoutKind.Sequential)> _
Public Structure VMR9AllocationInfo
	Public dwFlags As VMR9SurfaceAllocationFlags
	Public dwWidth As Integer
	Public dwHeight As Integer
    Public Format As Integer    ' D3DFORMAT
    Public Pool As Integer      ' D3DPOOL
	Public MinBuffers As Integer
	Public szAspectRatio As Size
	Public szNativeSize As Size
End Structure

<StructLayout(LayoutKind.Sequential)> _
Public Structure VMR9ProcAmpControl
    Public dwSize As Integer ' should be 24
	Public dwFlags As VMR9ProcAmpControlFlags
	Public Brightness As Single
	Public Contrast As Single
	Public Hue As Single
	Public Saturation As Single
End Structure

<StructLayout(LayoutKind.Sequential, CharSet := CharSet.Unicode)> _
Public Structure VMR9MonitorInfo
	Public uDevID As Integer
	Public rcMonitor As DsRect
	Public hMon As IntPtr
	Public dwFlags As Integer
	<MarshalAs(UnmanagedType.ByValTStr, SizeConst := 32)> _
	Public szDevice As String
	<MarshalAs(UnmanagedType.ByValTStr, SizeConst := 512)> _
	Public szDescription As String
	Public liDriverVersion As Long
	Public dwVendorId As Integer
	Public dwDeviceId As Integer
	Public dwSubSysId As Integer
	Public dwRevision As Integer
End Structure

<StructLayout(LayoutKind.Sequential)> _
Public Structure VMR9DeinterlaceCaps
	Public dwSize As Integer
	Public dwNumPreviousOutputFrames As Integer
	Public dwNumForwardRefSamples As Integer
	Public dwNumBackwardRefSamples As Integer
	Public DeinterlaceTechnology As VMR9DeinterlaceTech
End Structure

<StructLayout(LayoutKind.Sequential)> _
Public Structure VMR9VideoStreamInfo
    Public pddsVideoSurface As IntPtr  ' IDirect3DSurface9
	Public dwWidth As Integer
	Public dwHeight As Integer
	Public dwStrmID As Integer
	Public fAlpha As Single
	Public rNormal As NormalizedRect
	Public rtStart As Long
	Public rtEnd As Long
	Public SampleFormat As VMR9SampleFormat
End Structure

<StructLayout(LayoutKind.Sequential)> _
Public Structure VMR9VideoDesc
	Public dwSize As Integer
	Public dwSampleWidth As Integer
	Public dwSampleHeight As Integer
	Public SampleFormat As VMR9SampleFormat
	Public dwFourCC As Integer
	Public InputSampleFreq As VMR9Frequency
	Public OutputFrameFreq As VMR9Frequency
End Structure

<StructLayout(LayoutKind.Sequential)> _
Public Structure VMR9Frequency
	Public dwNumerator As Integer
	Public dwDenominator As Integer
End Structure

<StructLayout(LayoutKind.Sequential)> _
Public Structure VMR9AlphaBitmap
	Public dwFlags As VMR9AlphaBitmapFlags
    Public hdc As IntPtr    ' HDC
    Public pDDS As IntPtr   ' IDirect3DSurface9
	Public rSrc As DsRect
	Public rDest As NormalizedRect
	Public fAlpha As Single
	Public clrSrcKey As Integer
	Public dwFilterMode As VMRMixerPrefs
End Structure

<StructLayout(LayoutKind.Sequential)> _
Public Structure VMR9ProcAmpControlRange
    Public dwSize As Integer    ' should be 24
	Public dwProperty As VMR9ProcAmpControlFlags
	Public MinValue As Single
	Public MaxValue As Single
	Public DefaultValue As Single
	Public StepSize As Single
End Structure


#If ALLOW_UNTESTED_INTERFACES Then

<ComImport, System.Security.SuppressUnmanagedCodeSecurity, Guid("dfc581a1-6e1f-4c3a-8d0a-5e9792ea2afc"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)> _
Public Interface IVMRSurface9
	<PreserveSig> _
	Function IsSurfaceLocked() As Integer

	<PreserveSig> _
	Function LockSurface(<Out> ByRef lpSurface As IntPtr) As Integer ' BYTE**

	<PreserveSig> _
	Function UnlockSurface() As Integer

	<PreserveSig> _
	Function GetSurface(<Out, MarshalAs(UnmanagedType.IUnknown)> ByRef lplpSurface As Object) As Integer
End Interface

#End If

<ComImport, System.Security.SuppressUnmanagedCodeSecurity, Guid("69188c61-12a3-40f0-8ffc-342e7b433fd7"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)> _
Public Interface IVMRImagePresenter9
	<PreserveSig> _
	Function StartPresenting(<[In]> dwUserID As IntPtr) As Integer

	<PreserveSig> _
	Function StopPresenting(<[In]> dwUserID As IntPtr) As Integer

	<PreserveSig> _
	Function PresentImage(<[In]> dwUserID As IntPtr, <[In]> ByRef lpPresInfo As VMR9PresentationInfo) As Integer

End Interface

<ComImport, System.Security.SuppressUnmanagedCodeSecurity, Guid("6de9a68a-a928-4522-bf57-655ae3866456"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)> _
Public Interface IVMRSurfaceAllocatorEx9
	Inherits IVMRSurfaceAllocator9

    <PreserveSig()> _
    Shadows Function InitializeDevice(<[In]()> ByVal dwUserID As IntPtr, <[In]()> ByRef lpAllocInfo As VMR9AllocationInfo, <[In](), Out()> ByRef lpNumBuffers As Integer) As Integer

    <PreserveSig()> _
    Shadows Function TerminateDevice(<[In]()> ByVal dwID As IntPtr) As Integer

    <PreserveSig()> _
    Shadows Function GetSurface(<[In]()> ByVal dwUserID As IntPtr, <[In]()> ByVal SurfaceIndex As Integer, <[In]()> ByVal SurfaceFlags As Integer, <Out()> ByRef lplpSurface As IntPtr) As Integer

    <PreserveSig()> _
    Shadows Function AdviseNotify(<[In]()> ByVal lpIVMRSurfAllocNotify As IVMRSurfaceAllocatorNotify9) As Integer

	<PreserveSig> _
	Function GetSurfaceEx(<[In]> dwUserID As IntPtr, <[In]> SurfaceIndex As Integer, <[In]> SurfaceFlags As Integer, <Out> ByRef lplpSurface As IntPtr, <Out> ByRef lprcDst As DsRect) As Integer
End Interface

<ComImport, System.Security.SuppressUnmanagedCodeSecurity, Guid("dca3f5df-bb3a-4d03-bd81-84614bfbfa0c"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)> _
Public Interface IVMRSurfaceAllocatorNotify9
	<PreserveSig> _
	Function AdviseSurfaceAllocator(<[In]> dwUserID As IntPtr, <[In]> lpIVRMSurfaceAllocator As IVMRSurfaceAllocator9) As Integer

	<PreserveSig> _
	Function SetD3DDevice(<[In]> lpD3DDevice As IntPtr, <[In]> hMonitor As IntPtr) As Integer

	<PreserveSig> _
	Function ChangeD3DDevice(<[In]> lpD3DDevice As IntPtr, <[In]> hMonitor As IntPtr) As Integer

	<PreserveSig> _
	Function AllocateSurfaceHelper(<[In]> ByRef lpAllocInfo As VMR9AllocationInfo, <[In], Out> ByRef lpNumBuffers As Integer, <Out, MarshalAs(UnmanagedType.LPArray, ArraySubType := UnmanagedType.SysInt)> lplpSurface As IntPtr()) As Integer

	<PreserveSig> _
	Function NotifyEvent(<[In]> EvCode As EventCode, <[In]> Param1 As IntPtr, <[In]> Param2 As IntPtr) As Integer
End Interface

<ComImport, System.Security.SuppressUnmanagedCodeSecurity, Guid("8d5148ea-3f5d-46cf-9df1-d1b896eedb1f"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)> _
Public Interface IVMRSurfaceAllocator9
	<PreserveSig> _
	Function InitializeDevice(<[In]> dwUserID As IntPtr, <[In]> ByRef lpAllocInfo As VMR9AllocationInfo, <[In], Out> ByRef lpNumBuffers As Integer) As Integer

	<PreserveSig> _
	Function TerminateDevice(<[In]> dwID As IntPtr) As Integer

	<PreserveSig> _
	Function GetSurface(<[In]> dwUserID As IntPtr, <[In]> SurfaceIndex As Integer, <[In]> SurfaceFlags As Integer, <Out> ByRef lplpSurface As IntPtr) As Integer

	<PreserveSig> _
	Function AdviseNotify(<[In]> lpIVMRSurfAllocNotify As IVMRSurfaceAllocatorNotify9) As Integer
End Interface

<ComImport, System.Security.SuppressUnmanagedCodeSecurity, Guid("5a804648-4f66-4867-9c43-4f5c822cf1b8"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)> _
Public Interface IVMRFilterConfig9
	<PreserveSig> _
	Function SetImageCompositor(<[In]> lpVMRImgCompositor As IVMRImageCompositor9) As Integer

	<PreserveSig> _
	Function SetNumberOfStreams(<[In]> dwMaxStreams As Integer) As Integer

	<PreserveSig> _
	Function GetNumberOfStreams(<Out> ByRef pdwMaxStreams As Integer) As Integer

	<PreserveSig> _
	Function SetRenderingPrefs(<[In]> dwRenderFlags As VMR9RenderPrefs) As Integer

	<PreserveSig> _
	Function GetRenderingPrefs(<Out> ByRef pdwRenderFlags As VMR9RenderPrefs) As Integer

	<PreserveSig> _
	Function SetRenderingMode(<[In]> Mode As VMR9Mode) As Integer

	<PreserveSig> _
	Function GetRenderingMode(<Out> ByRef Mode As VMR9Mode) As Integer
End Interface

<ComImport, System.Security.SuppressUnmanagedCodeSecurity, Guid("8f537d09-f85e-4414-b23b-502e54c79927"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)> _
Public Interface IVMRWindowlessControl9
	Function GetNativeVideoSize(<Out> ByRef lpWidth As Integer, <Out> ByRef lpHeight As Integer, <Out> ByRef lpARWidth As Integer, <Out> ByRef lpARHeight As Integer) As Integer

	Function GetMinIdealVideoSize(<Out> ByRef lpWidth As Integer, <Out> ByRef lpHeight As Integer) As Integer

	Function GetMaxIdealVideoSize(<Out> ByRef lpWidth As Integer, <Out> ByRef lpHeight As Integer) As Integer

	Function SetVideoPosition(<[In]> lpSRCRect As DsRect, <[In]> lpDSTRect As DsRect) As Integer

	Function GetVideoPosition(<Out> lpSRCRect As DsRect, <Out> lpDSTRect As DsRect) As Integer

	Function GetAspectRatioMode(<Out> ByRef lpAspectRatioMode As VMR9AspectRatioMode) As Integer

	Function SetAspectRatioMode(<[In]> AspectRatioMode As VMR9AspectRatioMode) As Integer

	Function SetVideoClippingWindow(<[In]> hwnd As IntPtr) As Integer
	' HWND
	' HWND
    ' HDC
	Function RepaintVideo(<[In]> hwnd As IntPtr, <[In]> hdc As IntPtr) As Integer

	Function DisplayModeChanged() As Integer

	Function GetCurrentImage(<Out> ByRef lpDib As IntPtr) As Integer
	' BYTE**
	Function SetBorderColor(<[In]> Clr As Integer) As Integer

	Function GetBorderColor(<Out> ByRef lpClr As Integer) As Integer
End Interface

<ComImport, System.Security.SuppressUnmanagedCodeSecurity, Guid("00d96c29-bbde-4efc-9901-bb5036392146"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)> _
Public Interface IVMRAspectRatioControl9
	<PreserveSig> _
	Function GetAspectRatioMode(<Out> ByRef lpdwARMode As VMRAspectRatioMode) As Integer

	<PreserveSig> _
	Function SetAspectRatioMode(<[In]> lpdwARMode As VMRAspectRatioMode) As Integer
End Interface

<ComImport, System.Security.SuppressUnmanagedCodeSecurity, Guid("a215fb8d-13c2-4f7f-993c-003d6271a459"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)> _
Public Interface IVMRDeinterlaceControl9
	<PreserveSig> _
	Function GetNumberOfDeinterlaceModes(<[In]> ByRef lpVideoDescription As VMR9VideoDesc, <[In], Out> ByRef lpdwNumDeinterlaceModes As Integer, <Out, MarshalAs(UnmanagedType.LPArray, ArraySubType := UnmanagedType.Struct)> lpDeinterlaceModes As Guid()) As Integer

	<PreserveSig> _
	Function GetDeinterlaceModeCaps(<[In], MarshalAs(UnmanagedType.LPStruct)> lpDeinterlaceMode As Guid, <[In]> ByRef lpVideoDescription As VMR9VideoDesc, <[In], Out> ByRef lpDeinterlaceCaps As VMR9DeinterlaceCaps) As Integer

	<PreserveSig> _
	Function GetDeinterlaceMode(<[In]> dwStreamID As Integer, <Out> ByRef lpDeinterlaceMode As Guid) As Integer

	<PreserveSig> _
	Function SetDeinterlaceMode(<[In]> dwStreamID As Integer, <[In], MarshalAs(UnmanagedType.LPStruct)> lpDeinterlaceMode As Guid) As Integer

	<PreserveSig> _
	Function GetDeinterlacePrefs(<Out> ByRef lpdwDeinterlacePrefs As VMR9DeinterlacePrefs) As Integer

	<PreserveSig> _
	Function SetDeinterlacePrefs(<[In]> lpdwDeinterlacePrefs As VMR9DeinterlacePrefs) As Integer

	<PreserveSig> _
	Function GetActualDeinterlaceMode(<[In]> dwStreamID As Integer, <Out> ByRef lpDeinterlaceMode As Guid) As Integer
End Interface

<ComImport, System.Security.SuppressUnmanagedCodeSecurity, Guid("4a5c89eb-df51-4654-ac2a-e48e02bbabf6"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)> _
Public Interface IVMRImageCompositor9
	<PreserveSig> _
	Function InitCompositionDevice(<[In]> pD3DDevice As IntPtr) As Integer

	<PreserveSig> _
	Function TermCompositionDevice(<[In]> pD3DDevice As IntPtr) As Integer

	<PreserveSig> _
	Function SetStreamMediaType(<[In]> dwStrmID As Integer, <[In]> pmt As AMMediaType, <[In], MarshalAs(UnmanagedType.Bool)> fTexture As Boolean) As Integer

	' IDirect3DSurface9
	<PreserveSig> _
	Function CompositeImage(<[In]> pD3DDevice As IntPtr, <[In]> pddsRenderTarget As IntPtr, <[In]> pmtRenderTarget As AMMediaType, <[In]> rtStart As Long, <[In]> rtEnd As Long, <[In]> dwClrBkGnd As Integer, _
		<[In], MarshalAs(UnmanagedType.LPArray, ArraySubType := UnmanagedType.Struct, SizeParamIndex := 7)> pVideoStreamInfo As VMR9VideoStreamInfo(), <[In]> cStreams As Integer) As Integer
End Interface

<ComImport, System.Security.SuppressUnmanagedCodeSecurity, Guid("45c15cab-6e22-420a-8043-ae1f0ac02c7d"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)> _
Public Interface IVMRImagePresenterConfig9
	<PreserveSig> _
	Function SetRenderingPrefs(<[In]> dwRenderFlags As VMR9RenderPrefs) As Integer

	<PreserveSig> _
	Function GetRenderingPrefs(<Out> ByRef dwRenderFlags As VMR9RenderPrefs) As Integer
End Interface

<ComImport, System.Security.SuppressUnmanagedCodeSecurity, Guid("ced175e5-1935-4820-81bd-ff6ad00c9108"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)> _
Public Interface IVMRMixerBitmap9
	<PreserveSig> _
	Function SetAlphaBitmap(<[In]> ByRef pBmpParms As VMR9AlphaBitmap) As Integer

	<PreserveSig> _
	Function UpdateAlphaBitmapParameters(<[In]> ByRef pBmpParms As VMR9AlphaBitmap) As Integer

	<PreserveSig> _
	Function GetAlphaBitmapParameters(<Out> ByRef pBmpParms As VMR9AlphaBitmap) As Integer
End Interface

<ComImport, System.Security.SuppressUnmanagedCodeSecurity, Guid("1a777eaa-47c8-4930-b2c9-8fee1c1b0f3b"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)> _
Public Interface IVMRMixerControl9
	<PreserveSig> _
	Function SetAlpha(<[In]> dwStreamID As Integer, <[In]> Alpha As Single) As Integer

	<PreserveSig> _
	Function GetAlpha(<[In]> dwStreamID As Integer, <Out> ByRef Alpha As Single) As Integer

	<PreserveSig> _
	Function SetZOrder(<[In]> dwStreamID As Integer, <[In]> dwZ As Integer) As Integer

	<PreserveSig> _
	Function GetZOrder(<[In]> dwStreamID As Integer, <Out> ByRef dwZ As Integer) As Integer

	<PreserveSig> _
	Function SetOutputRect(<[In]> dwStreamID As Integer, <[In]> ByRef pRect As NormalizedRect) As Integer

	<PreserveSig> _
	Function GetOutputRect(<[In]> dwStreamID As Integer, <Out> ByRef pRect As NormalizedRect) As Integer

	<PreserveSig> _
	Function SetBackgroundClr(<[In]> ClrBkg As Integer) As Integer

	<PreserveSig> _
	Function GetBackgroundClr(<Out> ByRef ClrBkg As Integer) As Integer

	<PreserveSig> _
	Function SetMixingPrefs(<[In]> dwMixerPrefs As VMR9MixerPrefs) As Integer

	<PreserveSig> _
	Function GetMixingPrefs(<Out> ByRef dwMixerPrefs As VMR9MixerPrefs) As Integer

	<PreserveSig> _
	Function SetProcAmpControl(<[In]> dwStreamID As Integer, <[In]> ByRef lpClrControl As VMR9ProcAmpControl) As Integer

	<PreserveSig> _
	Function GetProcAmpControl(<[In]> dwStreamID As Integer, <[In], Out> ByRef lpClrControl As VMR9ProcAmpControl) As Integer

	<PreserveSig> _
	Function GetProcAmpControlRange(<[In]> dwStreamID As Integer, <[In], Out> ByRef lpClrControl As VMR9ProcAmpControlRange) As Integer
End Interface

<ComImport, System.Security.SuppressUnmanagedCodeSecurity, Guid("46c2e457-8ba0-4eef-b80b-0680f0978749"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)> _
Public Interface IVMRMonitorConfig9
	<PreserveSig> _
	Function SetMonitor(<[In]> uDev As Integer) As Integer

	<PreserveSig> _
	Function GetMonitor(<Out> ByRef uDev As Integer) As Integer

	<PreserveSig> _
	Function SetDefaultMonitor(<[In]> uDev As Integer) As Integer

	<PreserveSig> _
	Function GetDefaultMonitor(<Out> ByRef uDev As Integer) As Integer

	<PreserveSig> _
	Function GetAvailableMonitors(<Out, MarshalAs(UnmanagedType.LPArray, ArraySubType := UnmanagedType.Struct)> pInfo As VMR9MonitorInfo(), <[In]> dwMaxInfoArraySize As Integer, <Out> ByRef pdwNumDevices As Integer) As Integer
End Interface

<ComImport, System.Security.SuppressUnmanagedCodeSecurity, Guid("d0cfe38b-93e7-4772-8957-0400c49a4485"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)> _
Public Interface IVMRVideoStreamControl9
	<PreserveSig> _
	Function SetStreamActiveState(<[In], MarshalAs(UnmanagedType.Bool)> fActive As Boolean) As Integer

	<PreserveSig> _
	Function GetStreamActiveState(<Out, MarshalAs(UnmanagedType.Bool)> ByRef fActive As Boolean) As Integer
End Interface

