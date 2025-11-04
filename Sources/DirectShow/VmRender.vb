
Imports System.Drawing
Imports System.Runtime.InteropServices

#If ALLOW_UNTESTED_INTERFACES Then

<Flags> _
Public Enum VMRPresentationFlags
	None = 0
	SyncPoint = &H1
	Preroll = &H2
	Discontinuity = &H4
	TimeValid = &H8
	SrcDstRectsValid = &H10
End Enum

<Flags> _
Public Enum VMRSurfaceAllocationFlags
	None = 0
	PixelFormatValid = &H1
	ThreeDTarget = &H2
	AllowSysMem = &H4
	ForceSysMem = &H8
	DirectedFlip = &H10
	DXVATarget = &H20
End Enum

<StructLayout(LayoutKind.Sequential)> _
Public Structure VMRPresentationInfo
	Public dwFlags As VMRPresentationFlags
	Public lpSurf As IntPtr 'LPDIRECTDRAWSURFACE7
	Public rtStart As Long
	Public rtEnd As Long
	Public szAspectRatio As Size
	Public rcSrc As DsRect
	Public rcDst As DsRect
	Public dwTypeSpecificFlags As Integer
	Public dwInterlaceFlags As Integer
End Structure

<StructLayout(LayoutKind.Sequential)> _
Public Structure VMRAllocationInfo
	Public dwFlags As VMRSurfaceAllocationFlags
	'    public BitmapInfoHeader lpHdr;
	'    public DDPixelFormat lpPixFmt;
	Public lpHdr As IntPtr
	Public lpPixFmt As IntPtr
	Public szAspectRatio As Size
	Public dwMinBuffers As Integer
	Public dwMaxBuffers As Integer
	Public dwInterlaceFlags As Integer
	Public szNativeSize As Size
End Structure

#End If

<Flags> _
Public Enum VMRDeinterlaceTech
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
Public Enum VMRBitmap
	None = 0
	Disable = &H1
	Hdc = &H2
	EntireDDS = &H4
	SRCColorKey = &H8
	SRCRect = &H10
End Enum

<Flags> _
Public Enum VMRDeinterlacePrefs
	None = 0
	NextBest = &H1
	BOB = &H2
	Weave = &H4
	Mask = &H7
End Enum

<Flags> _
Public Enum VMRMixerPrefs
	None = 0
	NoDecimation = &H1
	DecimateOutput = &H2
	ARAdjustXorY = &H4
	DecimationReserved = &H8
	DecimateMask = &Hf

	BiLinearFiltering = &H10
	PointFiltering = &H20
	FilteringMask = &Hf0

	RenderTargetRGB = &H100
	RenderTargetYUV = &H1000

	RenderTargetYUV420 = &H200
	RenderTargetYUV422 = &H400
	RenderTargetYUV444 = &H800
	RenderTargetReserved = &He000
	RenderTargetMask = &Hff00

	DynamicSwitchToBOB = &H10000
	DynamicDecimateBy2 = &H20000

	DynamicReserved = &Hc0000
	DynamicMask = &Hf0000
End Enum

<Flags> _
Public Enum VMRRenderPrefs
	RestrictToInitialMonitor = &H0
	ForceOffscreen = &H1
	ForceOverlays = &H2
	AllowOverlays = &H0
	AllowOffscreen = &H0
	DoNotRenderColorKeyAndBorder = &H8
	Reserved = &H10
	PreferAGPMemWhenMixing = &H20

	Mask = &H3f
End Enum

<Flags> _
Public Enum VMRMode
	None = 0
	Windowed = &H1
	Windowless = &H2
	Renderless = &H4
End Enum

Public Enum VMRAspectRatioMode
	None
	LetterBox
End Enum

<StructLayout(LayoutKind.Sequential)> _
Public Structure VMRAlphaBitmap
	Public dwFlags As VMRBitmap
    Public hdc As IntPtr    ' HDC
    Public pDDS As IntPtr   'LPDIRECTDRAWSURFACE7
	Public rSrc As DsRect
	Public rDest As NormalizedRect
	Public fAlpha As Single
	Public clrSrcKey As Integer
End Structure

<StructLayout(LayoutKind.Sequential)> _
Public Structure VMRDeinterlaceCaps
	Public dwSize As Integer
	Public dwNumPreviousOutputFrames As Integer
	Public dwNumForwardRefSamples As Integer
	Public dwNumBackwardRefSamples As Integer
	Public DeinterlaceTechnology As VMRDeinterlaceTech
End Structure

<StructLayout(LayoutKind.Sequential)> _
Public Structure VMRFrequency
	Public dwNumerator As Integer
	Public dwDenominator As Integer
End Structure

<StructLayout(LayoutKind.Sequential, Pack := 1)> _
Public Structure VMRVideoDesc
	Public dwSize As Integer
	Public dwSampleWidth As Integer
	Public dwSampleHeight As Integer
	<MarshalAs(UnmanagedType.Bool)> _
	Public SingleFieldPerSample As Boolean
	Public dwFourCC As Integer
	Public InputSampleFreq As VMRFrequency
	Public OutputFrameFreq As VMRFrequency
End Structure

<StructLayout(LayoutKind.Sequential)> _
Public Structure VMRVideoStreamInfo
	Public pddsVideoSurface As IntPtr
	Public dwWidth As Integer
	Public dwHeight As Integer
	Public dwStrmID As Integer
	Public fAlpha As Single
	Public ddClrKey As DDColorKey
	Public rNormal As NormalizedRect
End Structure

<StructLayout(LayoutKind.Sequential)> _
Public Structure DDColorKey
	Public dw1 As Integer
	Public dw2 As Integer
End Structure

<StructLayout(LayoutKind.Sequential, CharSet := CharSet.Unicode)> _
Public Structure VMRMonitorInfo
	Public guid As VMRGuid
	Public rcMonitor As DsRect
    Public hMon As IntPtr       ' HMONITOR
	Public dwFlags As Integer
	<MarshalAs(UnmanagedType.ByValTStr, SizeConst := 32)> _
	Public szDevice As String
	<MarshalAs(UnmanagedType.ByValTStr, SizeConst := 256)> _
	Public szDescription As String
	Public liDriverVersion As Long
	Public dwVendorId As Integer
	Public dwDeviceId As Integer
	Public dwSubSysId As Integer
	Public dwRevision As Integer
End Structure

<StructLayout(LayoutKind.Sequential)> _
Public Structure VMRGuid
    Public pGUID As IntPtr  ' GUID *
	Public GUID As Guid
End Structure



#If ALLOW_UNTESTED_INTERFACES Then

<ComImport, System.Security.SuppressUnmanagedCodeSecurity, Guid("CE704FE7-E71E-41fb-BAA2-C4403E1182F5"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)> _
Public Interface IVMRImagePresenter
	<PreserveSig> _
	Function StartPresenting(<[In]> dwUserID As IntPtr) As Integer

	<PreserveSig> _
	Function StopPresenting(<[In]> dwUserID As IntPtr) As Integer

	<PreserveSig> _
	Function PresentImage(<[In]> dwUserID As IntPtr, <[In]> ByRef lpPresInfo As VMRPresentationInfo) As Integer
End Interface

<ComImport, System.Security.SuppressUnmanagedCodeSecurity, Guid("31ce832e-4484-458b-8cca-f4d7e3db0b52"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)> _
Public Interface IVMRSurfaceAllocator

    ' LPDIRECTDRAWSURFACE7
	<PreserveSig> _
	Function AllocateSurface(<[In]> dwUserID As IntPtr, <[In]> ByRef lpAllocInfo As VMRAllocationInfo, <Out> ByRef lpdwActualBuffers As Integer, <[In], Out> ByRef lplpSurface As IntPtr) As Integer

	<PreserveSig> _
	Function FreeSurface(<[In]> dwID As IntPtr) As Integer

	' LPDIRECTDRAWSURFACE7
	<PreserveSig> _
	Function PrepareSurface(<[In]> dwUserID As IntPtr, <[In]> lplpSurface As IntPtr, <[In]> dwSurfaceFlags As Integer) As Integer

	<PreserveSig> _
	Function AdviseNotify(<[In]> lpIVMRSurfAllocNotify As IVMRSurfaceAllocatorNotify) As Integer
End Interface

<ComImport, System.Security.SuppressUnmanagedCodeSecurity, Guid("aada05a8-5a4e-4729-af0b-cea27aed51e2"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)> _
Public Interface IVMRSurfaceAllocatorNotify
	<PreserveSig> _
	Function AdviseSurfaceAllocator(<[In]> dwUserID As IntPtr, <[In]> lpIVRMSurfaceAllocator As IVMRSurfaceAllocator) As Integer

	' LPDIRECTDRAW7
		' HMONITOR
	<PreserveSig> _
	Function SetDDrawDevice(<[In]> lpDDrawDevice As IntPtr, <[In]> hMonitor As IntPtr) As Integer

	' LPDIRECTDRAW7
		' HMONITOR
	<PreserveSig> _
	Function ChangeDDrawDevice(<[In]> lpDDrawDevice As IntPtr, <[In]> hMonitor As IntPtr) As Integer

	<PreserveSig> _
	Function RestoreDDrawSurfaces() As Integer

	<PreserveSig> _
	Function NotifyEvent(<[In]> EventCode As Integer, <[In]> Param1 As IntPtr, <[In]> Param2 As IntPtr) As Integer

	<PreserveSig> _
	Function SetBorderColor(<[In]> clrBorder As Integer) As Integer
End Interface

<ComImport, System.Security.SuppressUnmanagedCodeSecurity, Guid("a9849bbe-9ec8-4263-b764-62730f0d15d0"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)> _
Public Interface IVMRSurface
	<PreserveSig> _
	Function IsSurfaceLocked() As Integer

    ' BYTE**
	<PreserveSig> _
	Function LockSurface(<Out> ByRef lpSurface As IntPtr) As Integer

	<PreserveSig> _
	Function UnlockSurface() As Integer

	<PreserveSig> _
	Function GetSurface(<Out, MarshalAs(UnmanagedType.[Interface])> ByRef lplpSurface As Object) As Integer
End Interface

<ComImport, System.Security.SuppressUnmanagedCodeSecurity, Guid("e6f7ce40-4673-44f1-8f77-5499d68cb4ea"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)> _
Public Interface IVMRImagePresenterExclModeConfig
	Inherits IVMRImagePresenterConfig

    <PreserveSig()> _
    Shadows Function SetRenderingPrefs(<[In]()> ByVal dwRenderFlags As VMRRenderPrefs) As Integer

    <PreserveSig()> _
    Shadows Function GetRenderingPrefs(<Out()> ByRef dwRenderFlags As VMRRenderPrefs) As Integer

	<PreserveSig> _
	Function SetXlcModeDDObjAndPrimarySurface(<[In]> lpDDObj As IntPtr, <[In]> lpPrimarySurf As IntPtr) As Integer

	<PreserveSig> _
	Function GetXlcModeDDObjAndPrimarySurface(<Out> ByRef lpDDObj As IntPtr, <Out> ByRef lpPrimarySurf As IntPtr) As Integer
End Interface

#End If

<ComImport, System.Security.SuppressUnmanagedCodeSecurity, Guid("9e5530c5-7034-48b4-bb46-0b8a6efc8e36"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)> _
Public Interface IVMRFilterConfig
	<PreserveSig> _
	Function SetImageCompositor(<[In]> lpVMRImgCompositor As IVMRImageCompositor) As Integer

	<PreserveSig> _
	Function SetNumberOfStreams(<[In]> dwMaxStreams As Integer) As Integer

	<PreserveSig> _
	Function GetNumberOfStreams(<Out> ByRef pdwMaxStreams As Integer) As Integer

	<PreserveSig> _
	Function SetRenderingPrefs(<[In]> dwRenderFlags As VMRRenderPrefs) As Integer

	<PreserveSig> _
	Function GetRenderingPrefs(<Out> ByRef pdwRenderFlags As VMRRenderPrefs) As Integer

	<PreserveSig> _
	Function SetRenderingMode(<[In]> Mode As VMRMode) As Integer

	<PreserveSig> _
	Function GetRenderingMode(<Out> ByRef Mode As VMRMode) As Integer
End Interface

<ComImport, System.Security.SuppressUnmanagedCodeSecurity, Guid("0eb1088c-4dcd-46f0-878f-39dae86a51b7"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)> _
Public Interface IVMRWindowlessControl
	<PreserveSig> _
	Function GetNativeVideoSize(<Out> ByRef lpWidth As Integer, <Out> ByRef lpHeight As Integer, <Out> ByRef lpARWidth As Integer, <Out> ByRef lpARHeight As Integer) As Integer

	<PreserveSig> _
	Function GetMinIdealVideoSize(<Out> ByRef lpWidth As Integer, <Out> ByRef lpHeight As Integer) As Integer

	<PreserveSig> _
	Function GetMaxIdealVideoSize(<Out> ByRef lpWidth As Integer, <Out> ByRef lpHeight As Integer) As Integer

	<PreserveSig> _
	Function SetVideoPosition(<[In]> lpSRCRect As DsRect, <[In]> lpDSTRect As DsRect) As Integer

	<PreserveSig> _
	Function GetVideoPosition(<Out> lpSRCRect As DsRect, <Out> lpDSTRect As DsRect) As Integer

	<PreserveSig> _
	Function GetAspectRatioMode(<Out> ByRef lpAspectRatioMode As VMRAspectRatioMode) As Integer

	<PreserveSig> _
	Function SetAspectRatioMode(<[In]> AspectRatioMode As VMRAspectRatioMode) As Integer

	<PreserveSig> _
	Function SetVideoClippingWindow(<[In]> hwnd As IntPtr) As Integer

	<PreserveSig> _
	Function RepaintVideo(<[In]> hwnd As IntPtr, <[In]> hdc As IntPtr) As Integer

	<PreserveSig> _
	Function DisplayModeChanged() As Integer

    ' the caller is responsible for free the returned memory by calling CoTaskMemFree
    <PreserveSig()> _
    Function GetCurrentImage(<Out()> ByRef lpDib As IntPtr) As Integer ' BYTE**

	<PreserveSig> _
	Function SetBorderColor(<[In]> Clr As Integer) As Integer

	<PreserveSig> _
	Function GetBorderColor(<Out> ByRef lpClr As Integer) As Integer

	<PreserveSig> _
	Function SetColorKey(<[In]> Clr As Integer) As Integer

	<PreserveSig> _
	Function GetColorKey(<Out> ByRef lpClr As Integer) As Integer
End Interface

<ComImport, System.Security.SuppressUnmanagedCodeSecurity, Guid("ede80b5c-bad6-4623-b537-65586c9f8dfd"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)> _
Public Interface IVMRAspectRatioControl
	<PreserveSig> _
	Function GetAspectRatioMode(<Out> ByRef lpdwARMode As VMRAspectRatioMode) As Integer

	<PreserveSig> _
	Function SetAspectRatioMode(<[In]> lpdwARMode As VMRAspectRatioMode) As Integer
End Interface

<ComImport, System.Security.SuppressUnmanagedCodeSecurity, Guid("bb057577-0db8-4e6a-87a7-1a8c9a505a0f"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)> _
Public Interface IVMRDeinterlaceControl
	<PreserveSig> _
	Function GetNumberOfDeinterlaceModes(<[In]> ByRef lpVideoDescription As VMRVideoDesc, <[In], Out> ByRef lpdwNumDeinterlaceModes As Integer, <Out, MarshalAs(UnmanagedType.LPArray, ArraySubType := UnmanagedType.Struct)> lpDeinterlaceModes As Guid()) As Integer

	<PreserveSig> _
	Function GetDeinterlaceModeCaps(<[In], MarshalAs(UnmanagedType.LPStruct)> lpDeinterlaceMode As Guid, <[In]> ByRef lpVideoDescription As VMRVideoDesc, <[In], Out> ByRef lpDeinterlaceCaps As VMRDeinterlaceCaps) As Integer

	<PreserveSig> _
	Function GetDeinterlaceMode(<[In]> dwStreamID As Integer, <Out> ByRef lpDeinterlaceMode As Guid) As Integer

	<PreserveSig> _
	Function SetDeinterlaceMode(<[In]> dwStreamID As Integer, <[In], MarshalAs(UnmanagedType.LPStruct)> lpDeinterlaceMode As Guid) As Integer

	<PreserveSig> _
	Function GetDeinterlacePrefs(<Out> ByRef lpdwDeinterlacePrefs As VMRDeinterlacePrefs) As Integer

	<PreserveSig> _
	Function SetDeinterlacePrefs(<[In]> lpdwDeinterlacePrefs As VMRDeinterlacePrefs) As Integer

	<PreserveSig> _
	Function GetActualDeinterlaceMode(<[In]> dwStreamID As Integer, <Out> ByRef lpDeinterlaceMode As Guid) As Integer
End Interface

<ComImport, System.Security.SuppressUnmanagedCodeSecurity, Guid("7a4fb5af-479f-4074-bb40-ce6722e43c82"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)> _
Public Interface IVMRImageCompositor
	<PreserveSig> _
	Function InitCompositionTarget(<[In]> pD3DDevice As IntPtr, <[In]> pddsRenderTarget As IntPtr) As Integer

	<PreserveSig> _
	Function TermCompositionTarget(<[In]> pD3DDevice As IntPtr, <[In]> pddsRenderTarget As IntPtr) As Integer

	<PreserveSig> _
	Function SetStreamMediaType(<[In]> dwStrmID As Integer, <[In]> pmt As AMMediaType, <[In], MarshalAs(UnmanagedType.Bool)> fTexture As Boolean) As Integer

	<PreserveSig> _
	Function CompositeImage(<[In]> pD3DDevice As IntPtr, <[In]> pddsRenderTarget As IntPtr, <[In]> pmtRenderTarget As AMMediaType, <[In]> rtStart As Long, <[In]> rtEnd As Long, <[In]> dwClrBkGnd As Integer, _
		<[In], MarshalAs(UnmanagedType.LPArray, ArraySubType := UnmanagedType.Struct, SizeParamIndex := 7)> pVideoStreamInfo As VMRVideoStreamInfo(), <[In]> cStreams As Integer) As Integer
End Interface

<ComImport, System.Security.SuppressUnmanagedCodeSecurity, Guid("9f3a1c85-8555-49ba-935f-be5b5b29d178"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)> _
Public Interface IVMRImagePresenterConfig
	<PreserveSig> _
	Function SetRenderingPrefs(<[In]> dwRenderFlags As VMRRenderPrefs) As Integer

	<PreserveSig> _
	Function GetRenderingPrefs(<Out> ByRef dwRenderFlags As VMRRenderPrefs) As Integer
End Interface

<ComImport, System.Security.SuppressUnmanagedCodeSecurity, Guid("1E673275-0257-40aa-AF20-7C608D4A0428"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)> _
Public Interface IVMRMixerBitmap
	<PreserveSig> _
	Function SetAlphaBitmap(<[In]> ByRef pBmpParms As VMRAlphaBitmap) As Integer

	<PreserveSig> _
	Function UpdateAlphaBitmapParameters(<[In]> ByRef pBmpParms As VMRAlphaBitmap) As Integer

	<PreserveSig> _
	Function GetAlphaBitmapParameters(<Out> ByRef pBmpParms As VMRAlphaBitmap) As Integer
End Interface

<ComImport, System.Security.SuppressUnmanagedCodeSecurity, Guid("9cf0b1b6-fbaa-4b7f-88cf-cf1f130a0dce"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)> _
Public Interface IVMRMonitorConfig
	<PreserveSig> _
	Function SetMonitor(<[In]> ByRef pGUID As VMRGuid) As Integer

	<PreserveSig> _
	Function GetMonitor(<Out> ByRef pGUID As VMRGuid) As Integer

	<PreserveSig> _
	Function SetDefaultMonitor(<[In]> ByRef pGUID As VMRGuid) As Integer

	<PreserveSig> _
	Function GetDefaultMonitor(<Out> ByRef pGUID As VMRGuid) As Integer

	<PreserveSig> _
	Function GetAvailableMonitors(<Out, MarshalAs(UnmanagedType.LPArray, ArraySubType := UnmanagedType.Struct)> pInfo As VMRMonitorInfo(), <[In]> dwMaxInfoArraySize As Integer, <Out> ByRef pdwNumDevices As Integer) As Integer
End Interface

<ComImport, System.Security.SuppressUnmanagedCodeSecurity, Guid("058d1f11-2a54-4bef-bd54-df706626b727"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)> _
Public Interface IVMRVideoStreamControl
	<PreserveSig> _
	Function SetColorKey(<[In]> ByRef lpClrKey As DDColorKey) As Integer

	<PreserveSig> _
	Function GetColorKey(<Out> ByRef lpClrKey As DDColorKey) As Integer

	<PreserveSig> _
	Function SetStreamActiveState(<[In], MarshalAs(UnmanagedType.Bool)> fActive As Boolean) As Integer

	<PreserveSig> _
	Function GetStreamActiveState(<Out, MarshalAs(UnmanagedType.Bool)> ByRef fActive As Boolean) As Integer
End Interface

<ComImport, System.Security.SuppressUnmanagedCodeSecurity, Guid("1c1a17b0-bed0-415d-974b-dc6696131599"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)> _
Public Interface IVMRMixerControl
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
	Function SetMixingPrefs(<[In]> dwMixerPrefs As VMRMixerPrefs) As Integer

	<PreserveSig> _
	Function GetMixingPrefs(<Out> ByRef dwMixerPrefs As VMRMixerPrefs) As Integer
End Interface

<ComImport, System.Security.SuppressUnmanagedCodeSecurity, Guid("aac18c18-e186-46d2-825d-a1f8dc8e395a"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)> _
Public Interface IVPManager
	<PreserveSig> _
	Function SetVideoPortIndex(<[In]> dwVideoPortIndex As Integer) As Integer

	<PreserveSig> _
	Function GetVideoPortIndex(<Out> ByRef dwVideoPortIndex As Integer) As Integer
End Interface

