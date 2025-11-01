
Imports System.Drawing
Imports System.Runtime.InteropServices

#If ALLOW_UNTESTED_INTERFACES Then

Public Enum WSTStyle
	None = 0
	Invers
End Enum

<StructLayout(LayoutKind.Sequential)> _
Public Structure AMVABufferInfo
	Public dwTypeIndex As Integer
	Public dwBufferIndex As Integer
	Public dwDataOffset As Integer
	Public dwDataSize As Integer
End Structure

<StructLayout(LayoutKind.Sequential)> _
Public Structure AMVAUncompDataInfo
	Public dwUncompWidth As Integer
	Public dwUncompHeight As Integer
	Public ddUncompPixelFormat As DDPixelFormat
End Structure

<StructLayout(LayoutKind.Sequential)> _
Public Structure AMVAUncompBufferInfo
	Public dwMinNumSurfaces As Integer
	Public dwMaxNumSurfaces As Integer
	Public ddUncompPixelFormat As DDPixelFormat
End Structure

<StructLayout(LayoutKind.Sequential)> _
Public Structure AMVAInternalMemInfo
	Public dwScratchMemAlloc As Integer
End Structure

<StructLayout(LayoutKind.Sequential)> _
Public Structure DDSCaps2
	Public dwCaps As Integer
	Public dwCaps2 As Integer
	Public dwCaps3 As Integer
	Public dwCaps4 As Integer
End Structure

<StructLayout(LayoutKind.Sequential)> _
Public Structure AMVACompBufferInfo
	Public dwNumCompBuffers As Integer
	Public dwWidthToCreate As Integer
	Public dwHeightToCreate As Integer
	Public dwBytesToAllocate As Integer
	Public ddCompCaps As DDSCaps2
	Public ddPixelFormat As DDPixelFormat
End Structure

<StructLayout(LayoutKind.Sequential)> _
Public Structure AMVABeginFrameInfo
	Public dwDestSurfaceIndex As Integer
	Public pInputData As IntPtr  ' LPVOID
	Public dwSizeInputData As Integer
	Public pOutputData As IntPtr ' LPVOID
	Public dwSizeOutputData As Integer
End Structure

<StructLayout(LayoutKind.Sequential)> _
Public Structure AMVAEndFrameInfo
	Public dwSizeMiscData As Integer
	Public pMiscData As IntPtr ' LPVOID
End Structure

#End If

Public Enum WSTLevel
	Level1_5 = 0
End Enum

Public Enum WSTService
	None = 0
	Text
	IDS
	Invalid
End Enum

Public Enum WSTState
	Off = 0
	[On]
End Enum

Public Enum WSTDrawBGMode
	Opaque
	Transparent
End Enum

Public Enum MPEGAudioDivider
	CDAudio = 1
	FMRadio = 2
	AMRadio = 4
End Enum

Public Enum MPEGAudioDual
	Merge
	Left
	Right
End Enum

Public Enum MPEGAudioAccuracy
	Best = &H0
	High = &H4000
	Full = &H8000
End Enum

Public Enum MPEGAudioChannels
	Mono = 1
	Stereo = 2
End Enum

<StructLayout(LayoutKind.Sequential)> _
Public Structure WSTPage
	Public dwPageNr As Integer
	Public dwSubPageNr As Integer
    Public pucPageData As IntPtr  ' BYTE *
End Structure

<Flags> _
Public Enum AcmMpegHeadLayer As Short
	Layer1 = 1
	Layer2 = 2
	Layer3 = 4
End Enum

<Flags> _
Public Enum AcmMpegHeadMode As Short
	Stereo = 1
	JointStereo = 2
	DualChannel = 4
	SingleChannel = 8
End Enum

<Flags> _
Public Enum AcmMpegHeadFlags As Short
	None = &H0
	PrivateBit = &H1
	Copyright = &H2
	OriginalHome = &H4
	ProtectionBit = &H8
	IDMpeg1 = &H10
End Enum

<Flags> _
Public Enum WaveMask
	None = &H0
	FrontLeft = &H1
	FrontRight = &H2
	FrontCenter = &H4
	LowFrequency = &H8
	BackLeft = &H10
	BackRight = &H20
	FrontLeftOfCenter = &H40
	FrontRightOfCenter = &H80
	BackCenter = &H100
	SideLeft = &H200
	SideRight = &H400
	TopCenter = &H800
	TopFrontLeft = &H1000
	TopFrontCenter = &H2000
	TopFrontRight = &H4000
	TopBackLeft = &H8000
	TopBackCenter = &H10000
	TopBackRight = &H20000
End Enum

<StructLayout(LayoutKind.Sequential, Pack := 2)> _
Public Class MPEG1WaveFormat
	Public wfx As WaveFormatEx
	Public fwHeadLayer As AcmMpegHeadLayer
	Public dwHeadBitrate As Integer
	Public fwHeadMode As AcmMpegHeadMode
	Public fwHeadModeExt As Short
	Public wHeadEmphasis As Short
	Public fwHeadFlags As AcmMpegHeadFlags
	Public dwPTSLow As Integer
	Public dwPTSHigh As Integer
End Class

<StructLayout(LayoutKind.Explicit, Pack := 1)> _
Public Class WaveFormatExtensible
	Inherits WaveFormatEx
	<FieldOffset(0)> _
	Public wValidBitsPerSample As Short
	<FieldOffset(0)> _
	Public wSamplesPerBlock As Short
	<FieldOffset(0)> _
	Public wReserved As Short
	<FieldOffset(2)> _
	Public dwChannelMask As WaveMask
	<FieldOffset(6)> _
	Public SubFormat As Guid
End Class

Public Enum ASFWriterConfig
	None = 0
	AutoIndex = 1
	MultiPass = 2
	DontCompress = 3
End Enum


#If ALLOW_UNTESTED_INTERFACES Then

<ComImport, System.Security.SuppressUnmanagedCodeSecurity, Guid("c47a3420-005c-11d2-9038-00a0c9697298"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)> _
Public Interface IAMParse
	<PreserveSig> _
	Function GetParseTime(ByRef prtCurrent As Long) As Integer

	<PreserveSig> _
	Function SetParseTime(rtCurrent As Long) As Integer

	<PreserveSig> _
	Function Flush() As Integer
End Interface

<ComImport, System.Security.SuppressUnmanagedCodeSecurity, Guid("256A6A21-FBAD-11d1-82BF-00A0C9696C8F"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)> _
Public Interface IAMVideoAcceleratorNotify
	<PreserveSig> _
	Function GetUncompSurfacesInfo(<[In], MarshalAs(UnmanagedType.LPStruct)> pGuid As Guid, <Out> ByRef pUncompBufferInfo As AMVAUncompBufferInfo) As Integer

	<PreserveSig> _
	Function SetUncompSurfacesInfo(<[In]> dwActualUncompSurfacesAllocated As Integer) As Integer

	' LPVOID
	<PreserveSig> _
	Function GetCreateVideoAcceleratorData(<[In], MarshalAs(UnmanagedType.LPStruct)> pGuid As Guid, <Out> ByRef pdwSizeMiscData As Integer, <Out> ppMiscData As IntPtr) As Integer
End Interface

<ComImport, System.Security.SuppressUnmanagedCodeSecurity, Guid("256A6A22-FBAD-11d1-82BF-00A0C9696C8F"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)> _
Public Interface IAMVideoAccelerator
	<PreserveSig> _
	Function GetVideoAcceleratorGUIDs(<Out> ByRef pdwNumGuidsSupported As Integer, <[In], Out> pGuidsSupported As Guid()) As Integer

	<PreserveSig> _
	Function GetUncompFormatsSupported(<[In], MarshalAs(UnmanagedType.LPStruct)> pGuid As Guid, <Out> ByRef pdwNumFormatsSupported As Integer, <Out> ByRef pFormatsSupported As DDPixelFormat) As Integer

	<PreserveSig> _
	Function GetInternalMemInfo(<[In], MarshalAs(UnmanagedType.LPStruct)> pGuid As Guid, <[In]> pamvaUncompDataInfo As AMVAUncompDataInfo, <Out> ByRef pamvaInternalMemInfo As AMVAInternalMemInfo) As Integer

	<PreserveSig> _
	Function GetCompBufferInfo(<[In], MarshalAs(UnmanagedType.LPStruct)> pGuid As Guid, <[In]> pamvaUncompDataInfo As AMVAUncompDataInfo, <[In], Out> pdwNumTypesCompBuffers As Integer, <Out> ByRef pamvaCompBufferInfo As AMVACompBufferInfo) As Integer

	<PreserveSig> _
	Function GetInternalCompBufferInfo(<Out> ByRef pdwNumTypesCompBuffers As Integer, <Out> ByRef pamvaCompBufferInfo As AMVACompBufferInfo) As Integer

	<PreserveSig> _
	Function BeginFrame(<[In]> amvaBeginFrameInfo As AMVABeginFrameInfo) As Integer

	<PreserveSig> _
	Function EndFrame(<[In]> pEndFrameInfo As AMVAEndFrameInfo) As Integer

	' LPVOID
	<PreserveSig> _
	Function GetBuffer(<[In]> dwTypeIndex As Integer, <[In]> dwBufferIndex As Integer, <[In], MarshalAs(UnmanagedType.Bool)> bReadOnly As Boolean, <Out> ppBuffer As IntPtr, <Out> ByRef lpStride As Integer) As Integer

	<PreserveSig> _
	Function ReleaseBuffer(<[In]> dwTypeIndex As Integer, <[In]> dwBufferIndex As Integer) As Integer

	' LPVOID
	' LPVOID
	<PreserveSig> _
	Function Execute(<[In]> dwFunction As Integer, <[In]> lpPrivateInputData As IntPtr, <[In]> cbPrivateInputData As Integer, <[In]> lpPrivateOutputDat As IntPtr, <[In]> cbPrivateOutputData As Integer, <[In]> dwNumBuffers As Integer, _
		<[In]> pamvaBufferInfo As AMVABufferInfo) As Integer

	<PreserveSig> _
	Function QueryRenderStatus(<[In]> dwTypeIndex As Integer, <[In]> dwBufferIndex As Integer, <[In]> dwFlags As Integer) As Integer

	<PreserveSig> _
	Function DisplayFrame(<[In]> dwFlipToIndex As Integer, <[In]> pMediaSample As IMediaSample) As Integer
End Interface

<ComImport, System.Security.SuppressUnmanagedCodeSecurity, Guid("56a868fd-0ad4-11ce-b0a3-0020af0ba770"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)> _
Public Interface IAMFilterGraphCallback
	<PreserveSig> _
	Function UnableToRender(pPin As IPin) As Integer
End Interface

<ComImport, System.Security.SuppressUnmanagedCodeSecurity, Guid("AB6B4AFE-F6E4-11d0-900D-00C04FD9189D"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)> _
Public Interface IDirectDrawMediaSample
	' IDirectDrawSurface
	<PreserveSig> _
	Function GetSurfaceAndReleaseLock(<MarshalAs(UnmanagedType.IUnknown)> ByRef ppDirectDrawSurface As Object, ByRef pRect As Rectangle) As Integer

	<PreserveSig> _
	Function LockMediaSamplePointer() As Integer
End Interface

<ComImport, System.Security.SuppressUnmanagedCodeSecurity, Guid("AB6B4AFC-F6E4-11d0-900D-00C04FD9189D"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)> _
Public Interface IDirectDrawMediaSampleAllocator
	<PreserveSig> _
	Function GetDirectDraw(<MarshalAs(UnmanagedType.IUnknown)> ByRef ppDirectDraw As Object) As Integer
	' IDirectDraw
End Interface

#End If

<ComImport, System.Security.SuppressUnmanagedCodeSecurity, Guid("45086030-F7E4-486a-B504-826BB5792A3B"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)> _
Public Interface IConfigAsfWriter
	<PreserveSig, Obsolete("This method is now obsolete because it assumes version 4.0 Windows Media Format SDK profiles. Use GetCurrentProfile or GetCurrentProfileGuid instead to correctly identify a profile.", False)> _
	Function ConfigureFilterUsingProfileId(<[In]> dwProfileId As Integer) As Integer

	<PreserveSig, Obsolete("This method is now obsolete because it assumes version 4.0 Windows Media Format SDK profiles. Use GetCurrentProfile or GetCurrentProfileGuid instead to correctly identify a profile.", False)> _
	Function GetCurrentProfileId(<Out> ByRef pdwProfileId As Integer) As Integer

	<PreserveSig, Obsolete("Using Guids is considered obsolete by MS.  The preferred approach is using an IWMProfile.  See ConfigureFilterUsingProfile", False)> _
	Function ConfigureFilterUsingProfileGuid(<[In], MarshalAs(UnmanagedType.LPStruct)> guidProfile As Guid) As Integer

	<PreserveSig, Obsolete("Using Guids is considered obsolete by MS.  The preferred approach is using an IWMProfile.  See GetCurrentProfile", False)> _
	Function GetCurrentProfileGuid(<Out> ByRef pProfileGuid As Guid) As Integer

	<PreserveSig, Obsolete("This method requires IWMProfile, which in turn requires several other interfaces.  Rather than duplicate all those interfaces here, it is recommended that you use the WindowsMediaLib from http://DirectShowNet.SourceForge.net", False)> _
	Function ConfigureFilterUsingProfile(<[In]> pProfile As IntPtr) As Integer

	<PreserveSig, Obsolete("This method requires IWMProfile, which in turn requires several other interfaces.  Rather than duplicate all those interfaces here, it is recommended that you use the WindowsMediaLib from http://DirectShowNet.SourceForge.net", False)> _
	Function GetCurrentProfile(<Out> ByRef ppProfile As IntPtr) As Integer

	<PreserveSig> _
	Function SetIndexMode(<[In], MarshalAs(UnmanagedType.Bool)> bIndexFile As Boolean) As Integer

	<PreserveSig> _
	Function GetIndexMode(<Out, MarshalAs(UnmanagedType.Bool)> ByRef pbIndexFile As Boolean) As Integer
End Interface

<ComImport, System.Security.SuppressUnmanagedCodeSecurity, Guid("546F4260-D53E-11cf-B3F0-00AA003761C5"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)> _
Public Interface IAMDirectSound
	<PreserveSig> _
	Function GetDirectSoundInterface(<MarshalAs(UnmanagedType.IUnknown)> ByRef lplpds As Object) As Integer
	' IDirectSound
	<PreserveSig> _
	Function GetPrimaryBufferInterface(<MarshalAs(UnmanagedType.IUnknown)> ByRef lplpdsb As Object) As Integer
	' IDirectSoundBuffer
	<PreserveSig> _
	Function GetSecondaryBufferInterface(<MarshalAs(UnmanagedType.IUnknown)> ByRef lplpdsb As Object) As Integer
	' IDirectSoundBuffer
	<PreserveSig> _
	Function ReleaseDirectSoundInterface(<MarshalAs(UnmanagedType.IUnknown)> lpds As Object) As Integer
	' IDirectSound
	<PreserveSig> _
	Function ReleasePrimaryBufferInterface(<MarshalAs(UnmanagedType.IUnknown)> lpdsb As Object) As Integer
	' IDirectSoundBuffer
	<PreserveSig> _
	Function ReleaseSecondaryBufferInterface(<MarshalAs(UnmanagedType.IUnknown)> lpdsb As Object) As Integer
	' IDirectSoundBuffer
	<PreserveSig> _
	Function SetFocusWindow(hWnd As IntPtr, <[In], MarshalAs(UnmanagedType.Bool)> bSet As Boolean) As Integer

	<PreserveSig> _
	Function GetFocusWindow(ByRef hWnd As IntPtr, <Out, MarshalAs(UnmanagedType.Bool)> ByRef bSet As Boolean) As Integer
End Interface

<ComImport, System.Security.SuppressUnmanagedCodeSecurity, Guid("C056DE21-75C2-11d3-A184-00105AEF9F33"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)> _
Public Interface IAMWstDecoder
	<PreserveSig> _
	Function GetDecoderLevel(ByRef lpLevel As WSTLevel) As Integer

	<PreserveSig> _
	Function GetCurrentService(ByRef lpService As WSTService) As Integer

	<PreserveSig> _
	Function GetServiceState(ByRef lpState As WSTState) As Integer

	<PreserveSig> _
	Function SetServiceState(State As WSTState) As Integer

	<PreserveSig> _
	Function GetOutputFormat(<MarshalAs(UnmanagedType.LPStruct)> lpbmih As BitmapInfoHeader) As Integer

	<PreserveSig> _
	Function SetOutputFormat(lpbmi As BitmapInfoHeader) As Integer

	<PreserveSig> _
	Function GetBackgroundColor(ByRef pdwPhysColor As Integer) As Integer

	<PreserveSig> _
	Function SetBackgroundColor(dwPhysColor As Integer) As Integer

	<PreserveSig> _
	Function GetRedrawAlways(<MarshalAs(UnmanagedType.Bool)> ByRef lpbOption As Boolean) As Integer

	<PreserveSig> _
	Function SetRedrawAlways(<MarshalAs(UnmanagedType.Bool)> bOption As Boolean) As Integer

	<PreserveSig> _
	Function GetDrawBackgroundMode(ByRef lpMode As WSTDrawBGMode) As Integer

	<PreserveSig> _
	Function SetDrawBackgroundMode(Mode As WSTDrawBGMode) As Integer

	<PreserveSig> _
	Function SetAnswerMode(<MarshalAs(UnmanagedType.Bool)> bAnswer As Boolean) As Integer

	<PreserveSig> _
	Function GetAnswerMode(<MarshalAs(UnmanagedType.Bool)> ByRef pbAnswer As Boolean) As Integer

	<PreserveSig> _
	Function SetHoldPage(<MarshalAs(UnmanagedType.Bool)> bHoldPage As Boolean) As Integer

	<PreserveSig> _
	Function GetHoldPage(<MarshalAs(UnmanagedType.Bool)> ByRef pbHoldPage As Boolean) As Integer

	<PreserveSig> _
	Function GetCurrentPage(<[In], Out> pWstPage As WSTPage) As Integer

	<PreserveSig> _
	Function SetCurrentPage(<[In]> WstPage As WSTPage) As Integer
End Interface

<ComImport, System.Security.SuppressUnmanagedCodeSecurity, Guid("b45dd570-3c77-11d1-abe1-00a0c905f375"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)> _
Public Interface IMpegAudioDecoder
	<PreserveSig> _
	Function get_FrequencyDivider(ByRef pDivider As MPEGAudioDivider) As Integer

	<PreserveSig> _
	Function put_FrequencyDivider(Divider As MPEGAudioDivider) As Integer

	<PreserveSig> _
	Function get_DecoderAccuracy(ByRef pAccuracy As MPEGAudioAccuracy) As Integer

	<PreserveSig> _
	Function put_DecoderAccuracy(Accuracy As MPEGAudioAccuracy) As Integer

	<PreserveSig> _
	Function get_Stereo(ByRef pStereo As MPEGAudioChannels) As Integer

	<PreserveSig> _
	Function put_Stereo(Stereo As MPEGAudioChannels) As Integer

	<PreserveSig> _
	Function get_DecoderWordSize(ByRef pWordSize As Integer) As Integer

	<PreserveSig> _
	Function put_DecoderWordSize(WordSize As Integer) As Integer

	<PreserveSig> _
	Function get_IntegerDecode(ByRef pIntDecode As Integer) As Integer

	<PreserveSig> _
	Function put_IntegerDecode(IntDecode As Integer) As Integer

	<PreserveSig> _
	Function get_DualMode(ByRef pIntDecode As MPEGAudioDual) As Integer

	<PreserveSig> _
	Function put_DualMode(IntDecode As MPEGAudioDual) As Integer

	<PreserveSig> _
	Function get_AudioFormat(ByRef lpFmt As MPEG1WaveFormat) As Integer
End Interface

'<ComImport, System.Security.SuppressUnmanagedCodeSecurity, Guid("6d5140c1-7436-11ce-8034-00aa006009fa"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)> _
'Public Interface IServiceProvider
'	<PreserveSig> _
'	Function QueryService(<[In], MarshalAs(UnmanagedType.LPStruct)> guidService As DsGuid, <[In], MarshalAs(UnmanagedType.LPStruct)> riid As DsGuid, <MarshalAs(UnmanagedType.IUnknown)> ByRef ppvObject As Object) As Integer
'End Interface

<ComImport, System.Security.SuppressUnmanagedCodeSecurity, Guid("FC4801A3-2BA9-11CF-A229-00AA003D7352"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)> _
Public Interface IObjectWithSite
	<PreserveSig> _
	Function SetSite(<[In], MarshalAs(UnmanagedType.IUnknown)> pUnkSite As Object) As Integer

	<PreserveSig> _
	Function GetSite(<[In], MarshalAs(UnmanagedType.LPStruct)> riid As DsGuid, <MarshalAs(UnmanagedType.IUnknown)> ByRef ppvSite As Object) As Integer
End Interface

<ComImport, System.Security.SuppressUnmanagedCodeSecurity, Guid("7989CCAA-53F0-44f0-884A-F3B03F6AE066"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)> _
Public Interface IConfigAsfWriter2
	Inherits IConfigAsfWriter

    <PreserveSig(), Obsolete("This method is now obsolete because it assumes version 4.0 Windows Media Format SDK profiles. Use GetCurrentProfile or GetCurrentProfileGuid instead to correctly identify a profile.", False)> _
    Shadows Function ConfigureFilterUsingProfileId(<[In]()> ByVal dwProfileId As Integer) As Integer

    <PreserveSig(), Obsolete("This method is now obsolete because it assumes version 4.0 Windows Media Format SDK profiles. Use GetCurrentProfile or GetCurrentProfileGuid instead to correctly identify a profile.", False)> _
    Shadows Function GetCurrentProfileId(<Out()> ByRef pdwProfileId As Integer) As Integer

    <PreserveSig(), Obsolete("Using Guids is considered obsolete by MS.  The preferred approach is using an IWMProfile.  See ConfigureFilterUsingProfile", False)> _
    Shadows Function ConfigureFilterUsingProfileGuid(<[In](), MarshalAs(UnmanagedType.LPStruct)> ByVal guidProfile As Guid) As Integer

    <PreserveSig(), Obsolete("Using Guids is considered obsolete by MS.  The preferred approach is using an IWMProfile.  See GetCurrentProfile", False)> _
    Shadows Function GetCurrentProfileGuid(<Out()> ByRef pProfileGuid As Guid) As Integer

    <PreserveSig(), Obsolete("This method requires IWMProfile, which in turn requires several other interfaces.  Rather than duplicate all those interfaces here, it is recommended that you use the WindowsMediaLib from http://DirectShowNet.SourceForge.net", False)> _
    Shadows Function ConfigureFilterUsingProfile(<[In]()> ByVal pProfile As IntPtr) As Integer

    <PreserveSig(), Obsolete("This method requires IWMProfile, which in turn requires several other interfaces.  Rather than duplicate all those interfaces here, it is recommended that you use the WindowsMediaLib from http://DirectShowNet.SourceForge.net", False)> _
    Shadows Function GetCurrentProfile(<Out()> ByRef ppProfile As IntPtr) As Integer

    <PreserveSig()> _
    Shadows Function SetIndexMode(<[In](), MarshalAs(UnmanagedType.Bool)> ByVal bIndexFile As Boolean) As Integer

    <PreserveSig()> _
    Shadows Function GetIndexMode(<Out(), MarshalAs(UnmanagedType.Bool)> ByRef pbIndexFile As Boolean) As Integer

	<PreserveSig> _
	Function StreamNumFromPin(pPin As IPin, ByRef pwStreamNum As Short) As Integer

	<PreserveSig> _
	Function SetParam(dwParam As ASFWriterConfig, dwParam1 As Integer, dwParam2 As Integer) As Integer

	<PreserveSig> _
	Function GetParam(dwParam As ASFWriterConfig, ByRef pdwParam1 As Integer, pdwParam2 As IntPtr) As Integer

	<PreserveSig> _
	Function ResetMultiPassState() As Integer

End Interface

<ComImport, System.Security.SuppressUnmanagedCodeSecurity, Guid("a8809222-07bb-48ea-951c-33158100625b"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)> _
Public Interface IGetCapabilitiesKey
    <PreserveSig()> _
    Function GetCapabilitiesKey(<Out()> ByRef pHKey As IntPtr) As Integer  ' HKEY
End Interface

