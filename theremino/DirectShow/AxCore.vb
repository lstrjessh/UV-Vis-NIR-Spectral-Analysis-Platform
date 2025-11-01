
Imports System.Runtime.InteropServices

<Flags> _
Public Enum AMGBF
	None = 0
	PrevFrameSkipped = 1
	NotAsyncPoint = 2
	NoWait = 4
	NoDDSurfaceLock = 8
End Enum

<Flags> _
Public Enum AMVideoFlag
	FieldMask = &H3
	InterleavedFrame = &H0
	Field1 = &H1
	Field2 = &H2
	Field1First = &H4
	Weave = &H8
	IPBMask = &H30
	ISample = &H0
	PSample = &H10
	BSample = &H20
	RepeatField = &H40
End Enum

<Flags> _
Public Enum AMSamplePropertyFlags
	SplicePoint = &H1
	PreRoll = &H2
	DataDiscontinuity = &H4
	TypeChanged = &H8
	TimeValid = &H10
	MediaTimeValid = &H20
	TimeDiscontinuity = &H40
	FlushOnPause = &H80
	StopValid = &H100
	EndOfStream = &H200
	Media = 0
	Control = 1
End Enum

<StructLayout(LayoutKind.Sequential, CharSet := CharSet.Unicode)> _
Public Structure PinInfo
	<MarshalAs(UnmanagedType.[Interface])> _
	Public filter As IBaseFilter
	Public dir As PinDirection
	<MarshalAs(UnmanagedType.ByValTStr, SizeConst := 128)> _
	Public name As String
End Structure

' When you are done with an instance of this class,
' it should be released with FreeAMMediaType() to avoid leaking
<StructLayout(LayoutKind.Sequential)> _
Public Class AMMediaType
	Public majorType As Guid
	Public subType As Guid
	<MarshalAs(UnmanagedType.Bool)> _
	Public fixedSizeSamples As Boolean
	<MarshalAs(UnmanagedType.Bool)> _
	Public temporalCompression As Boolean
	Public sampleSize As Integer
	Public formatType As Guid
    Public unkPtr As IntPtr         ' IUnknown Pointer
	Public formatSize As Integer
    Public formatPtr As IntPtr      ' Pointer to a buff determined by formatType
End Class

Public Enum PinDirection
	Input
	Output
End Enum

<Flags> _
Public Enum AMSeekingSeekingCapabilities
	None = 0
	CanSeekAbsolute = &H1
	CanSeekForwards = &H2
	CanSeekBackwards = &H4
	CanGetCurrentPos = &H8
	CanGetStopPos = &H10
	CanGetDuration = &H20
	CanPlayBackwards = &H40
	CanDoSegments = &H80
	Source = &H100
End Enum

Public Enum FilterState
	Stopped
	Paused
	Running
End Enum

<StructLayout(LayoutKind.Sequential, CharSet := CharSet.Unicode)> _
Public Structure FilterInfo
	<MarshalAs(UnmanagedType.ByValTStr, SizeConst := 128)> _
	Public achName As String
	<MarshalAs(UnmanagedType.[Interface])> _
	Public pGraph As IFilterGraph
End Structure

<Flags> _
Public Enum AMSeekingSeekingFlags
	NoPositioning = &H0
	AbsolutePositioning = &H1
	RelativePositioning = &H2
	IncrementalPositioning = &H3
	PositioningBitsMask = &H3
	SeekToKeyFrame = &H4
	ReturnTime = &H8
	Segment = &H10
	NoFlush = &H20
End Enum

<StructLayout(LayoutKind.Sequential)> _
Public Class AllocatorProperties
	Public cBuffers As Integer
	Public cbBuffer As Integer
	Public cbAlign As Integer
	Public cbPrefix As Integer
End Class

<StructLayout(LayoutKind.Sequential)> _
Public Class AMSample2Properties
	Public cbData As Integer
	Public dwTypeSpecificFlags As AMVideoFlag
	Public dwSampleFlags As AMSamplePropertyFlags
	Public lActual As Integer
	Public tStart As Long
	Public tStop As Long
	Public dwStreamId As Integer
	Public pMediaType As IntPtr
    Public pbBuffer As IntPtr   ' BYTE *
	Public cbBuffer As Integer
End Class


#If ALLOW_UNTESTED_INTERFACES Then

<ComImport, System.Security.SuppressUnmanagedCodeSecurity, Guid("68961E68-832B-41ea-BC91-63593F3E70E3"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)> _
Public Interface IMediaSample2Config
	<PreserveSig> _
	Function GetSurface(<MarshalAs(UnmanagedType.IUnknown)> ByRef ppDirect3DSurface9 As Object) As Integer

End Interface

<ComImport, System.Security.SuppressUnmanagedCodeSecurity, Guid("36b73885-c2c8-11cf-8b46-00805f6cef60"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)> _
Public Interface IReferenceClock2
	Inherits IReferenceClock

    <PreserveSig()> _
    Shadows Function GetTime(<Out()> ByRef pTime As Long) As Integer

    ' System.Threading.WaitHandle?
    <PreserveSig()> _
    Shadows Function AdviseTime(<[In]()> ByVal baseTime As Long, <[In]()> ByVal streamTime As Long, <[In]()> ByVal hEvent As IntPtr, <Out()> ByRef pdwAdviseCookie As Integer) As Integer

    ' System.Threading.WaitHandle?
    <PreserveSig()> _
    Shadows Function AdvisePeriodic(<[In]()> ByVal startTime As Long, <[In]()> ByVal periodTime As Long, <[In]()> ByVal hSemaphore As IntPtr, <Out()> ByRef pdwAdviseCookie As Integer) As Integer

    <PreserveSig()> _
    Shadows Function Unadvise(<[In]()> ByVal dwAdviseCookie As Integer) As Integer
End Interface

<ComImport, System.Security.SuppressUnmanagedCodeSecurity, Guid("56a8689d-0ad4-11ce-b03a-0020af0ba770"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)> _
Public Interface IMemInputPin
	<PreserveSig> _
	Function GetAllocator(<Out> ByRef ppAllocator As IMemAllocator) As Integer

	<PreserveSig> _
	Function NotifyAllocator(<[In]> pAllocator As IMemAllocator, <[In], MarshalAs(UnmanagedType.Bool)> bReadOnly As Boolean) As Integer

	<PreserveSig> _
	Function GetAllocatorRequirements(<Out> ByRef pProps As AllocatorProperties) As Integer

	<PreserveSig> _
	Function Receive(<[In]> pSample As IMediaSample) As Integer

	' IMediaSample[]
	<PreserveSig> _
	Function ReceiveMultiple(<[In]> pSamples As IntPtr, <[In]> nSamples As Integer, <Out> ByRef nSamplesProcessed As Integer) As Integer

	<PreserveSig> _
	Function ReceiveCanBlock() As Integer
End Interface

<ComImport, System.Security.SuppressUnmanagedCodeSecurity, Guid("a3d8cec0-7e5a-11cf-bbc5-00805f6cef20"), Obsolete("This interface has been deprecated.", False), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)> _
Public Interface IAMovieSetup
	<PreserveSig> _
	Function Register() As Integer

	<PreserveSig> _
	Function Unregister() As Integer
End Interface

#End If

<ComImport, System.Security.SuppressUnmanagedCodeSecurity, Guid("56a86891-0ad4-11ce-b03a-0020af0ba770"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)> _
Public Interface IPin
	<PreserveSig> _
	Function Connect(<[In]> pReceivePin As IPin, <[In], MarshalAs(UnmanagedType.LPStruct)> pmt As AMMediaType) As Integer

	<PreserveSig> _
	Function ReceiveConnection(<[In]> pReceivePin As IPin, <[In], MarshalAs(UnmanagedType.LPStruct)> pmt As AMMediaType) As Integer

	<PreserveSig> _
	Function Disconnect() As Integer

	<PreserveSig> _
	Function ConnectedTo(<Out> ByRef ppPin As IPin) As Integer

    ' Release returned parameter with DsUtils.FreeAMMediaType
	<PreserveSig> _
	Function ConnectionMediaType(<Out, MarshalAs(UnmanagedType.LPStruct)> pmt As AMMediaType) As Integer

    ' Release returned parameter with DsUtils.FreePinInfo
	<PreserveSig> _
	Function QueryPinInfo(<Out> ByRef pInfo As PinInfo) As Integer

	<PreserveSig> _
	Function QueryDirection(ByRef pPinDir As PinDirection) As Integer

	<PreserveSig> _
	Function QueryId(<Out, MarshalAs(UnmanagedType.LPWStr)> ByRef Id As String) As Integer

	<PreserveSig> _
	Function QueryAccept(<[In], MarshalAs(UnmanagedType.LPStruct)> pmt As AMMediaType) As Integer

	<PreserveSig> _
	Function EnumMediaTypes(<Out> ByRef ppEnum As IEnumMediaTypes) As Integer

	<PreserveSig> _
	Function QueryInternalConnections(<Out, MarshalAs(UnmanagedType.LPArray, SizeParamIndex := 1)> ppPins As IPin(), <[In], Out> ByRef nPin As Integer) As Integer

	<PreserveSig> _
	Function EndOfStream() As Integer

	<PreserveSig> _
	Function BeginFlush() As Integer

	<PreserveSig> _
	Function EndFlush() As Integer

	<PreserveSig> _
	Function NewSegment(<[In]> tStart As Long, <[In]> tStop As Long, <[In]> dRate As Double) As Integer
End Interface

<ComImport, System.Security.SuppressUnmanagedCodeSecurity, Guid("36b73880-c2c8-11cf-8b46-00805f6cef60"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)> _
Public Interface IMediaSeeking
	<PreserveSig> _
	Function GetCapabilities(<Out> ByRef pCapabilities As AMSeekingSeekingCapabilities) As Integer

	<PreserveSig> _
	Function CheckCapabilities(<[In], Out> ByRef pCapabilities As AMSeekingSeekingCapabilities) As Integer

	<PreserveSig> _
	Function IsFormatSupported(<[In], MarshalAs(UnmanagedType.LPStruct)> pFormat As Guid) As Integer

	<PreserveSig> _
	Function QueryPreferredFormat(<Out> ByRef pFormat As Guid) As Integer

	<PreserveSig> _
	Function GetTimeFormat(<Out> ByRef pFormat As Guid) As Integer

	<PreserveSig> _
	Function IsUsingTimeFormat(<[In], MarshalAs(UnmanagedType.LPStruct)> pFormat As Guid) As Integer

	<PreserveSig> _
	Function SetTimeFormat(<[In], MarshalAs(UnmanagedType.LPStruct)> pFormat As Guid) As Integer

	<PreserveSig> _
	Function GetDuration(<Out> ByRef pDuration As Long) As Integer

	<PreserveSig> _
	Function GetStopPosition(<Out> ByRef pStop As Long) As Integer

	<PreserveSig> _
	Function GetCurrentPosition(<Out> ByRef pCurrent As Long) As Integer

	<PreserveSig> _
	Function ConvertTimeFormat(<Out> ByRef pTarget As Long, <[In], MarshalAs(UnmanagedType.LPStruct)> pTargetFormat As DsGuid, <[In]> Source As Long, <[In], MarshalAs(UnmanagedType.LPStruct)> pSourceFormat As DsGuid) As Integer

	<PreserveSig> _
	Function SetPositions(<[In], Out, MarshalAs(UnmanagedType.LPStruct)> pCurrent As DsLong, <[In]> dwCurrentFlags As AMSeekingSeekingFlags, <[In], Out, MarshalAs(UnmanagedType.LPStruct)> pStop As DsLong, <[In]> dwStopFlags As AMSeekingSeekingFlags) As Integer

	<PreserveSig> _
	Function GetPositions(<Out> ByRef pCurrent As Long, <Out> ByRef pStop As Long) As Integer

	<PreserveSig> _
	Function GetAvailable(<Out> ByRef pEarliest As Long, <Out> ByRef pLatest As Long) As Integer

	<PreserveSig> _
	Function SetRate(<[In]> dRate As Double) As Integer

	<PreserveSig> _
	Function GetRate(<Out> ByRef pdRate As Double) As Integer

	<PreserveSig> _
	Function GetPreroll(<Out> ByRef pllPreroll As Long) As Integer
End Interface

<ComImport, System.Security.SuppressUnmanagedCodeSecurity, Guid("56a8689a-0ad4-11ce-b03a-0020af0ba770"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)> _
Public Interface IMediaSample
    <PreserveSig()> _
    Function GetPointer(<Out()> ByRef ppBuffer As IntPtr) As Integer  ' BYTE **

	<PreserveSig> _
	Function GetSize() As Integer

	<PreserveSig> _
	Function GetTime(<Out> ByRef pTimeStart As Long, <Out> ByRef pTimeEnd As Long) As Integer

	<PreserveSig> _
	Function SetTime(<[In], MarshalAs(UnmanagedType.LPStruct)> pTimeStart As DsLong, <[In], MarshalAs(UnmanagedType.LPStruct)> pTimeEnd As DsLong) As Integer

	<PreserveSig> _
	Function IsSyncPoint() As Integer

	<PreserveSig> _
	Function SetSyncPoint(<[In], MarshalAs(UnmanagedType.Bool)> bIsSyncPoint As Boolean) As Integer

	<PreserveSig> _
	Function IsPreroll() As Integer

	<PreserveSig> _
	Function SetPreroll(<[In], MarshalAs(UnmanagedType.Bool)> bIsPreroll As Boolean) As Integer

	<PreserveSig> _
	Function GetActualDataLength() As Integer

	<PreserveSig> _
	Function SetActualDataLength(<[In]> len As Integer) As Integer

    ' Returned object must be released with DsUtils.FreeAMMediaType()
	<PreserveSig> _
	Function GetMediaType(<Out, MarshalAs(UnmanagedType.LPStruct)> ByRef ppMediaType As AMMediaType) As Integer

	<PreserveSig> _
	Function SetMediaType(<[In], MarshalAs(UnmanagedType.LPStruct)> pMediaType As AMMediaType) As Integer

	<PreserveSig> _
	Function IsDiscontinuity() As Integer

	<PreserveSig> _
	Function SetDiscontinuity(<[In], MarshalAs(UnmanagedType.Bool)> bDiscontinuity As Boolean) As Integer

	<PreserveSig> _
	Function GetMediaTime(<Out> ByRef pTimeStart As Long, <Out> ByRef pTimeEnd As Long) As Integer

	<PreserveSig> _
	Function SetMediaTime(<[In], MarshalAs(UnmanagedType.LPStruct)> pTimeStart As DsLong, <[In], MarshalAs(UnmanagedType.LPStruct)> pTimeEnd As DsLong) As Integer
End Interface

<ComImport, System.Security.SuppressUnmanagedCodeSecurity, Guid("56a86899-0ad4-11ce-b03a-0020af0ba770"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)> _
Public Interface IMediaFilter
	Inherits IPersist

    <PreserveSig()> _
    Shadows Function GetClassID(<Out()> ByRef pClassID As Guid) As Integer

	<PreserveSig> _
	Function [Stop]() As Integer

	<PreserveSig> _
	Function Pause() As Integer

	<PreserveSig> _
	Function Run(<[In]> tStart As Long) As Integer

	<PreserveSig> _
	Function GetState(<[In]> dwMilliSecsTimeout As Integer, <Out> ByRef filtState As FilterState) As Integer

	<PreserveSig> _
	Function SetSyncSource(<[In]> pClock As IReferenceClock) As Integer

	<PreserveSig> _
	Function GetSyncSource(<Out> ByRef pClock As IReferenceClock) As Integer
End Interface

<ComImport, System.Security.SuppressUnmanagedCodeSecurity, Guid("56a86895-0ad4-11ce-b03a-0020af0ba770"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)> _
Public Interface IBaseFilter
	Inherits IMediaFilter

    <PreserveSig()> _
    Shadows Function GetClassID(<Out()> ByRef pClassID As Guid) As Integer

    <PreserveSig()> _
    Shadows Function [Stop]() As Integer

    <PreserveSig()> _
    Shadows Function Pause() As Integer

    <PreserveSig()> _
    Shadows Function Run(ByVal tStart As Long) As Integer

    <PreserveSig()> _
    Shadows Function GetState(<[In]()> ByVal dwMilliSecsTimeout As Integer, <Out()> ByRef filtState As FilterState) As Integer

    <PreserveSig()> _
    Shadows Function SetSyncSource(<[In]()> ByVal pClock As IReferenceClock) As Integer

    <PreserveSig()> _
    Shadows Function GetSyncSource(<Out()> ByRef pClock As IReferenceClock) As Integer

    <PreserveSig()> _
    Function EnumPins(<Out()> ByRef ppEnum As IEnumPins) As Integer

    <PreserveSig()> _
    Function FindPin(<[In](), MarshalAs(UnmanagedType.LPWStr)> ByVal Id As String, <Out()> ByRef ppPin As IPin) As Integer

    <PreserveSig()> _
    Function QueryFilterInfo(<Out()> ByRef pInfo As FilterInfo) As Integer

    <PreserveSig()> _
    Function JoinFilterGraph(<[In]()> ByVal pGraph As IFilterGraph, <[In](), MarshalAs(UnmanagedType.LPWStr)> ByVal pName As String) As Integer

    <PreserveSig()> _
    Function QueryVendorInfo(<Out(), MarshalAs(UnmanagedType.LPWStr)> ByRef pVendorInfo As String) As Integer
End Interface

<ComImport(), System.Security.SuppressUnmanagedCodeSecurity(), Guid("56a8689f-0ad4-11ce-b03a-0020af0ba770"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)> _
Public Interface IFilterGraph
    <PreserveSig()> _
    Function AddFilter(<[In]()> ByVal pFilter As IBaseFilter, <[In](), MarshalAs(UnmanagedType.LPWStr)> ByVal pName As String) As Integer

    <PreserveSig()> _
    Function RemoveFilter(<[In]()> ByVal pFilter As IBaseFilter) As Integer

    <PreserveSig()> _
    Function EnumFilters(<Out()> ByRef ppEnum As IEnumFilters) As Integer

    <PreserveSig()> _
    Function FindFilterByName(<[In](), MarshalAs(UnmanagedType.LPWStr)> ByVal pName As String, <Out()> ByRef ppFilter As IBaseFilter) As Integer

    <PreserveSig()> _
    Function ConnectDirect(<[In]()> ByVal ppinOut As IPin, <[In]()> ByVal ppinIn As IPin, <[In](), MarshalAs(UnmanagedType.LPStruct)> ByVal pmt As AMMediaType) As Integer

    <PreserveSig()> _
    <Obsolete("This method is obsolete; use the IFilterGraph2.ReconnectEx method instead.")> _
    Function Reconnect(<[In]()> ByVal ppin As IPin) As Integer

    <PreserveSig()> _
    Function Disconnect(<[In]()> ByVal ppin As IPin) As Integer

    <PreserveSig()> _
    Function SetDefaultSyncSource() As Integer
End Interface

<ComImport(), System.Security.SuppressUnmanagedCodeSecurity(), Guid("56a86893-0ad4-11ce-b03a-0020af0ba770"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)> _
Public Interface IEnumFilters
    <PreserveSig()> _
    Function [Next](<[In]()> ByVal cFilters As Integer, <Out(), MarshalAs(UnmanagedType.LPArray, SizeParamIndex:=0)> ByVal ppFilter As IBaseFilter(), <[In]()> ByVal pcFetched As IntPtr) As Integer

    <PreserveSig()> _
    Function Skip(<[In]()> ByVal cFilters As Integer) As Integer

    <PreserveSig()> _
    Function Reset() As Integer

    <PreserveSig()> _
    Function Clone(<Out()> ByRef ppEnum As IEnumFilters) As Integer
End Interface

<ComImport(), System.Security.SuppressUnmanagedCodeSecurity(), Guid("56a86892-0ad4-11ce-b03a-0020af0ba770"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)> _
Public Interface IEnumPins
    <PreserveSig()> _
    Function [Next](<[In]()> ByVal cPins As Integer, <Out(), MarshalAs(UnmanagedType.LPArray, SizeParamIndex:=0)> ByVal ppPins As IPin(), <[In]()> ByVal pcFetched As IntPtr) As Integer

    <PreserveSig()> _
    Function Skip(<[In]()> ByVal cPins As Integer) As Integer

    <PreserveSig()> _
    Function Reset() As Integer

    <PreserveSig()> _
    Function Clone(<Out()> ByRef ppEnum As IEnumPins) As Integer
End Interface

<ComImport(), System.Security.SuppressUnmanagedCodeSecurity(), Guid("56a86897-0ad4-11ce-b03a-0020af0ba770"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)> _
Public Interface IReferenceClock
    <PreserveSig()> _
    Function GetTime(<Out()> ByRef pTime As Long) As Integer

    ' System.Threading.WaitHandle?
    <PreserveSig()> _
    Function AdviseTime(<[In]()> ByVal baseTime As Long, <[In]()> ByVal streamTime As Long, <[In]()> ByVal hEvent As IntPtr, <Out()> ByRef pdwAdviseCookie As Integer) As Integer

    ' System.Threading.WaitHandle?
    <PreserveSig()> _
    Function AdvisePeriodic(<[In]()> ByVal startTime As Long, <[In]()> ByVal periodTime As Long, <[In]()> ByVal hSemaphore As IntPtr, <Out()> ByRef pdwAdviseCookie As Integer) As Integer

    <PreserveSig()> _
    Function Unadvise(<[In]()> ByVal dwAdviseCookie As Integer) As Integer
End Interface

<ComImport(), System.Security.SuppressUnmanagedCodeSecurity(), Guid("89c31040-846b-11ce-97d3-00aa0055595a"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)> _
Public Interface IEnumMediaTypes
    <PreserveSig()> _
    Function [Next](<[In]()> ByVal cMediaTypes As Integer, <[In](), Out(), MarshalAs(UnmanagedType.CustomMarshaler, MarshalTypeRef:=GetType(EMTMarshaler), SizeParamIndex:=0)> ByVal ppMediaTypes As AMMediaType(), <[In]()> ByVal pcFetched As IntPtr) As Integer

    <PreserveSig()> _
    Function Skip(<[In]()> ByVal cMediaTypes As Integer) As Integer

    <PreserveSig()> _
    Function Reset() As Integer

    <PreserveSig()> _
    Function Clone(<Out()> ByRef ppEnum As IEnumMediaTypes) As Integer
End Interface

<ComImport(), System.Security.SuppressUnmanagedCodeSecurity(), Guid("36b73884-c2c8-11cf-8b46-00805f6cef60"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)> _
Public Interface IMediaSample2
    Inherits IMediaSample

    <PreserveSig()> _
    Shadows Function GetPointer(<Out()> ByRef ppBuffer As IntPtr) As Integer
    ' BYTE **
    <PreserveSig()> _
    Shadows Function GetSize() As Integer

    <PreserveSig()> _
    Shadows Function GetTime(<Out()> ByRef pTimeStart As Long, <Out()> ByRef pTimeEnd As Long) As Integer

    <PreserveSig()> _
    Shadows Function SetTime(<[In](), MarshalAs(UnmanagedType.LPStruct)> ByVal pTimeStart As DsLong, <[In](), MarshalAs(UnmanagedType.LPStruct)> ByVal pTimeEnd As DsLong) As Integer

    <PreserveSig()> _
    Shadows Function IsSyncPoint() As Integer

    <PreserveSig()> _
    Shadows Function SetSyncPoint(<[In](), MarshalAs(UnmanagedType.Bool)> ByVal bIsSyncPoint As Boolean) As Integer

    <PreserveSig()> _
    Shadows Function IsPreroll() As Integer

    <PreserveSig()> _
    Shadows Function SetPreroll(<[In](), MarshalAs(UnmanagedType.Bool)> ByVal bIsPreroll As Boolean) As Integer

    <PreserveSig()> _
    Shadows Function GetActualDataLength() As Integer

    <PreserveSig()> _
    Shadows Function SetActualDataLength(<[In]()> ByVal len As Integer) As Integer

    <PreserveSig()> _
    Shadows Function GetMediaType(<Out()> ByRef ppMediaType As AMMediaType) As Integer

    <PreserveSig()> _
    Shadows Function SetMediaType(<[In]()> ByVal pMediaType As AMMediaType) As Integer

    <PreserveSig()> _
    Shadows Function IsDiscontinuity() As Integer

    <PreserveSig()> _
    Shadows Function SetDiscontinuity(<[In](), MarshalAs(UnmanagedType.Bool)> ByVal bDiscontinuity As Boolean) As Integer

    <PreserveSig()> _
    Shadows Function GetMediaTime(<Out()> ByRef pTimeStart As Long, <Out()> ByRef pTimeEnd As Long) As Integer

    <PreserveSig()> _
    Shadows Function SetMediaTime(<[In](), MarshalAs(UnmanagedType.LPStruct)> ByVal pTimeStart As DsLong, <[In](), MarshalAs(UnmanagedType.LPStruct)> ByVal pTimeEnd As DsLong) As Integer

    ' BYTE *
    <PreserveSig()> _
    Function GetProperties(<[In]()> ByVal cbProperties As Integer, <[In]()> ByVal pbProperties As IntPtr) As Integer

    ' BYTE *
    <PreserveSig()> _
    Function SetProperties(<[In]()> ByVal cbProperties As Integer, <[In]()> ByVal pbProperties As IntPtr) As Integer
End Interface

<ComImport(), System.Security.SuppressUnmanagedCodeSecurity(), Guid("92980b30-c1de-11d2-abf5-00a0c905f375"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)> _
Public Interface IMemAllocatorNotifyCallbackTemp
    <PreserveSig()> _
    Function NotifyRelease() As Integer
End Interface

<ComImport(), System.Security.SuppressUnmanagedCodeSecurity(), Guid("379a0cf0-c1de-11d2-abf5-00a0c905f375"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)> _
Public Interface IMemAllocatorCallbackTemp
    Inherits IMemAllocator

    <PreserveSig()> _
    Shadows Function SetProperties(<[In]()> ByVal pRequest As AllocatorProperties, <Out(), MarshalAs(UnmanagedType.LPStruct)> ByVal pActual As AllocatorProperties) As Integer

    <PreserveSig()> _
    Shadows Function GetProperties(<Out()> ByVal pProps As AllocatorProperties) As Integer

    <PreserveSig()> _
    Shadows Function Commit() As Integer

    <PreserveSig()> _
    Shadows Function Decommit() As Integer

    <PreserveSig()> _
    Shadows Function GetBuffer(<Out()> ByRef ppBuffer As IMediaSample, <[In]()> ByVal pStartTime As Long, <[In]()> ByVal pEndTime As Long, <[In]()> ByVal dwFlags As AMGBF) As Integer

    <PreserveSig()> _
    Shadows Function ReleaseBuffer(<[In]()> ByVal pBuffer As IMediaSample) As Integer

    <PreserveSig()> _
    Function SetNotify(<[In]()> ByVal pNotify As IMemAllocatorNotifyCallbackTemp) As Integer

    <PreserveSig()> _
    Function GetFreeCount(<Out()> ByRef plBuffersFree As Integer) As Integer
End Interface

<ComImport(), System.Security.SuppressUnmanagedCodeSecurity(), Guid("56a8689c-0ad4-11ce-b03a-0020af0ba770"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)> _
Public Interface IMemAllocator
    <PreserveSig()> _
    Function SetProperties(<[In](), MarshalAs(UnmanagedType.LPStruct)> ByVal pRequest As AllocatorProperties, <Out(), MarshalAs(UnmanagedType.LPStruct)> ByVal pActual As AllocatorProperties) As Integer

    <PreserveSig()> _
    Function GetProperties(<Out(), MarshalAs(UnmanagedType.LPStruct)> ByVal pProps As AllocatorProperties) As Integer

    <PreserveSig()> _
    Function Commit() As Integer

    <PreserveSig()> _
    Function Decommit() As Integer

    <PreserveSig()> _
    Function GetBuffer(<Out()> ByRef ppBuffer As IMediaSample, <[In]()> ByVal pStartTime As Long, <[In]()> ByVal pEndTime As Long, <[In]()> ByVal dwFlags As AMGBF) As Integer

    <PreserveSig()> _
    Function ReleaseBuffer(<[In]()> ByVal pBuffer As IMediaSample) As Integer
End Interface

<ComImport(), System.Security.SuppressUnmanagedCodeSecurity(), Guid("ebec459c-2eca-4d42-a8af-30df557614b8"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)> _
Public Interface IReferenceClockTimerControl
    <PreserveSig()> _
    Function SetDefaultTimerResolution(ByVal timerResolution As Long) As Integer

    <PreserveSig()> _
    Function GetDefaultTimerResolution(ByRef pTimerResolution As Long) As Integer
End Interface

