
Imports System.Runtime.InteropServices
Imports System.Runtime.InteropServices.ComTypes

Namespace MultimediaStreaming

    <Flags()> _
    Public Enum AMMStream
        None = &H0
        AddDefaultRenderer = &H1
        CreatePeer = &H2
        StopIfNoSamples = &H4
        NoStall = &H8
    End Enum

    <Flags()> _
    Public Enum AMMMultiStream
        None = &H0
        NoGraphThread = &H1
    End Enum

    <Flags()> _
    Public Enum AMOpenModes
        RenderTypeMask = &H3
        RenderToExisting = 0
        RenderAllStreams = &H1
        NoRender = &H2
        NoClock = &H4
        Run = &H8
    End Enum


    <ComImport(), System.Security.SuppressUnmanagedCodeSecurity(), Guid("BEBE595D-9A6F-11D0-8FDE-00C04FD9189D"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)> _
    Public Interface IAMMediaStream
        Inherits IMediaStream

        <PreserveSig()> _
        Shadows Function GetMultiMediaStream(<MarshalAs(UnmanagedType.[Interface])> ByRef ppMultiMediaStream As IMultiMediaStream) As Integer

        <PreserveSig()> _
        Shadows Function GetInformation(ByRef pPurposeId As Guid, ByRef pType As StreamType) As Integer

        <PreserveSig()> _
        Shadows Function SetSameFormat(<[In](), MarshalAs(UnmanagedType.[Interface])> ByVal pStreamThatHasDesiredFormat As IMediaStream, <[In]()> ByVal dwFlags As Integer) As Integer

        <PreserveSig()> _
        Shadows Function AllocateSample(<[In]()> ByVal dwFlags As Integer, <MarshalAs(UnmanagedType.[Interface])> ByRef ppSample As IStreamSample) As Integer

        <PreserveSig()> _
        Shadows Function CreateSharedSample(<[In](), MarshalAs(UnmanagedType.[Interface])> ByVal pExistingSample As IStreamSample, <[In]()> ByVal dwFlags As Integer, <MarshalAs(UnmanagedType.[Interface])> ByRef ppNewSample As IStreamSample) As Integer

        <PreserveSig()> _
        Shadows Function SendEndOfStream(ByVal dwFlags As Integer) As Integer

        <PreserveSig()> _
        Function Initialize(<[In](), MarshalAs(UnmanagedType.IUnknown)> ByVal pSourceObject As Object, <[In]()> ByVal dwFlags As AMMStream, <[In](), MarshalAs(UnmanagedType.LPStruct)> ByVal PurposeId As Guid, <[In]()> ByVal StreamType As StreamType) As Integer

        <PreserveSig()> _
        Function SetState(<[In]()> ByVal State As FilterState) As Integer

        <PreserveSig()> _
        Function JoinAMMultiMediaStream(<[In](), MarshalAs(UnmanagedType.[Interface])> ByVal pAMMultiMediaStream As IAMMultiMediaStream) As Integer

        <PreserveSig()> _
        Function JoinFilter(<[In](), MarshalAs(UnmanagedType.[Interface])> ByVal pMediaStreamFilter As IMediaStreamFilter) As Integer

        <PreserveSig()> _
        Function JoinFilterGraph(<[In](), MarshalAs(UnmanagedType.[Interface])> ByVal pFilterGraph As IFilterGraph) As Integer
    End Interface

    <ComImport(), System.Security.SuppressUnmanagedCodeSecurity(), InterfaceType(ComInterfaceType.InterfaceIsIUnknown), Guid("BEBE595C-9A6F-11D0-8FDE-00C04FD9189D")> _
    Public Interface IAMMultiMediaStream
        Inherits IMultiMediaStream

        <PreserveSig()> _
        Shadows Function GetInformation(ByRef pdwFlags As MMSSF, ByRef pStreamType As StreamType) As Integer

        <PreserveSig()> _
        Shadows Function GetMediaStream(<[In](), MarshalAs(UnmanagedType.LPStruct)> ByVal idPurpose As Guid, <MarshalAs(UnmanagedType.[Interface])> ByRef ppMediaStream As IMediaStream) As Integer

        <PreserveSig()> _
        Shadows Function EnumMediaStreams(<[In]()> ByVal Index As Integer, <MarshalAs(UnmanagedType.[Interface])> ByRef ppMediaStream As IMediaStream) As Integer

        <PreserveSig()> _
        Shadows Function GetState(ByRef pCurrentState As StreamState) As Integer

        <PreserveSig()> _
        Shadows Function SetState(<[In]()> ByVal NewState As StreamState) As Integer

        <PreserveSig()> _
        Shadows Function GetTime(ByRef pCurrentTime As Long) As Integer

        <PreserveSig()> _
        Shadows Function GetDuration(ByRef pDuration As Long) As Integer

        <PreserveSig()> _
        Shadows Function Seek(<[In]()> ByVal SeekTime As Long) As Integer

        <PreserveSig()> _
        Shadows Function GetEndOfStreamEventHandle(ByRef phEOS As IntPtr) As Integer

        <PreserveSig()> _
        Function Initialize(<[In]()> ByVal StreamType As StreamType, <[In]()> ByVal dwFlags As AMMMultiStream, <[In](), MarshalAs(UnmanagedType.[Interface])> ByVal pFilterGraph As IGraphBuilder) As Integer

        <PreserveSig()> _
        Function GetFilterGraph(<MarshalAs(UnmanagedType.[Interface])> ByRef ppGraphBuilder As IGraphBuilder) As Integer

        <PreserveSig()> _
        Function GetFilter(<MarshalAs(UnmanagedType.[Interface])> ByRef ppFilter As IMediaStreamFilter) As Integer

        <PreserveSig()> _
        Function AddMediaStream(<[In](), MarshalAs(UnmanagedType.IUnknown)> ByVal pStreamObject As Object, <[In]()> ByVal PurposeId As DsGuid, <[In]()> ByVal dwFlags As AMMStream, <Out()> ByVal ppNewStream As IMediaStream) As Integer

        <PreserveSig()> _
        Function OpenFile(<[In](), MarshalAs(UnmanagedType.LPWStr)> ByVal pszFileName As String, <[In]()> ByVal dwFlags As AMOpenModes) As Integer

        <PreserveSig()> _
        Function OpenMoniker(<[In](), MarshalAs(UnmanagedType.[Interface])> ByVal pCtx As IBindCtx, <[In](), MarshalAs(UnmanagedType.[Interface])> ByVal pMoniker As IMoniker, <[In]()> ByVal dwFlags As AMOpenModes) As Integer

        <PreserveSig()> _
        Function Render(<[In]()> ByVal dwFlags As AMOpenModes) As Integer
    End Interface

    <ComImport(), System.Security.SuppressUnmanagedCodeSecurity(), InterfaceType(ComInterfaceType.InterfaceIsIUnknown), Guid("AB6B4AFA-F6E4-11D0-900D-00C04FD9189D")> _
    Public Interface IAMMediaTypeStream
        Inherits IMediaStream

        <PreserveSig()> _
        Shadows Function GetMultiMediaStream(<MarshalAs(UnmanagedType.[Interface])> ByRef ppMultiMediaStream As IMultiMediaStream) As Integer

        <PreserveSig()> _
        Shadows Function GetInformation(ByRef pPurposeId As Guid, ByRef pType As StreamType) As Integer

        <PreserveSig()> _
        Shadows Function SetSameFormat(<[In](), MarshalAs(UnmanagedType.[Interface])> ByVal pStreamThatHasDesiredFormat As IMediaStream, <[In]()> ByVal dwFlags As Integer) As Integer

        <PreserveSig()> _
        Shadows Function AllocateSample(<[In]()> ByVal dwFlags As Integer, <MarshalAs(UnmanagedType.[Interface])> ByRef ppSample As IStreamSample) As Integer

        <PreserveSig()> _
        Shadows Function CreateSharedSample(<[In](), MarshalAs(UnmanagedType.[Interface])> ByVal pExistingSample As IStreamSample, <[In]()> ByVal dwFlags As Integer, <MarshalAs(UnmanagedType.[Interface])> ByRef ppNewSample As IStreamSample) As Integer

        <PreserveSig()> _
        Shadows Function SendEndOfStream(ByVal dwFlags As Integer) As Integer

        <PreserveSig()> _
        Function GetFormat(<Out(), MarshalAs(UnmanagedType.LPStruct)> ByVal pMediaType As AMMediaType, <[In]()> ByVal dwFlags As Integer) As Integer

        <PreserveSig()> _
        Function SetFormat(<[In](), MarshalAs(UnmanagedType.LPStruct)> ByVal pMediaType As AMMediaType, <[In]()> ByVal dwFlags As Integer) As Integer

        <PreserveSig()> _
        Function CreateSample(<[In]()> ByVal lSampleSize As Integer, <[In]()> ByVal pbBuffer As IntPtr, <[In]()> ByVal dwFlags As Integer, <[In](), MarshalAs(UnmanagedType.IUnknown)> ByVal pUnkOuter As Object, <MarshalAs(UnmanagedType.[Interface])> ByRef ppAMMediaTypeSample As IAMMediaTypeSample) As Integer

        <PreserveSig()> _
        Function GetStreamAllocatorRequirements(ByRef pProps As AllocatorProperties) As Integer

        <PreserveSig()> _
        Function SetStreamAllocatorRequirements(<[In](), MarshalAs(UnmanagedType.LPStruct)> ByVal pProps As AllocatorProperties) As Integer
    End Interface

    <ComImport(), System.Security.SuppressUnmanagedCodeSecurity(), InterfaceType(ComInterfaceType.InterfaceIsIUnknown), Guid("AB6B4AFB-F6E4-11D0-900D-00C04FD9189D")> _
    Public Interface IAMMediaTypeSample
        Inherits IStreamSample

        <PreserveSig()> _
        Shadows Function GetMediaStream(<MarshalAs(UnmanagedType.[Interface])> ByRef ppMediaStream As IMediaStream) As Integer

        <PreserveSig()> _
        Shadows Function GetSampleTimes(ByRef pStartTime As Long, ByRef pEndTime As Long, ByRef pCurrentTime As Long) As Integer

        <PreserveSig()> _
        Shadows Function SetSampleTimes(<[In]()> ByVal pStartTime As DsLong, <[In]()> ByVal pEndTime As DsLong) As Integer

        <PreserveSig()> _
        Shadows Function Update(<[In]()> ByVal dwFlags As SSUpdate, <[In]()> ByVal hEvent As IntPtr, <[In]()> ByVal pfnAPC As IntPtr, <[In]()> ByVal dwAPCData As IntPtr) As Integer

        <PreserveSig()> _
        Shadows Function CompletionStatus(<[In]()> ByVal dwFlags As CompletionStatusFlags, <[In]()> ByVal dwMilliseconds As Integer) As Integer

        <PreserveSig()> _
        Function SetPointer(<[In]()> ByVal pBuffer As IntPtr, <[In]()> ByVal lSize As Integer) As Integer

        <PreserveSig()> _
        Function GetPointer(<Out()> ByRef ppBuffer As IntPtr) As Integer

        <PreserveSig()> _
        Function GetSize() As Integer

        <PreserveSig()> _
        Function GetTime(ByRef pTimeStart As Long, ByRef pTimeEnd As Long) As Integer

        <PreserveSig()> _
        Function SetTime(<[In]()> ByVal pTimeStart As DsLong, <[In]()> ByVal pTimeEnd As DsLong) As Integer

        <PreserveSig()> _
        Function IsSyncPoint() As Integer

        <PreserveSig()> _
        Function SetSyncPoint(<[In](), MarshalAs(UnmanagedType.Bool)> ByVal IsSyncPoint As Boolean) As Integer

        <PreserveSig()> _
        Function IsPreroll() As Integer

        <PreserveSig()> _
        Function SetPreroll(<[In](), MarshalAs(UnmanagedType.Bool)> ByVal IsPreroll As Boolean) As Integer

        <PreserveSig()> _
        Function GetActualDataLength() As Integer

        <PreserveSig()> _
        Function SetActualDataLength(ByVal Size As Integer) As Integer

        <PreserveSig()> _
        Function GetMediaType(ByRef ppMediaType As AMMediaType) As Integer

        <PreserveSig()> _
        Function SetMediaType(<[In](), MarshalAs(UnmanagedType.LPStruct)> ByVal pMediaType As AMMediaType) As Integer

        <PreserveSig()> _
        Function IsDiscontinuity() As Integer

        <PreserveSig()> _
        Function SetDiscontinuity(<[In](), MarshalAs(UnmanagedType.Bool)> ByVal Discontinuity As Boolean) As Integer

        <PreserveSig()> _
        Function GetMediaTime(ByRef pTimeStart As Long, ByRef pTimeEnd As Long) As Integer

        <PreserveSig()> _
        Function SetMediaTime(<[In]()> ByVal pTimeStart As DsLong, <[In]()> ByVal pTimeEnd As DsLong) As Integer
    End Interface

    <ComImport(), System.Security.SuppressUnmanagedCodeSecurity(), Guid("BEBE595E-9A6F-11D0-8FDE-00C04FD9189D"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)> _
    Public Interface IMediaStreamFilter
        Inherits IBaseFilter

        <PreserveSig()> _
        Shadows Function GetClassID(ByRef pClassID As Guid) As Integer

        <PreserveSig()> _
        Shadows Function [Stop]() As Integer

        <PreserveSig()> _
        Shadows Function Pause() As Integer

        <PreserveSig()> _
        Shadows Function Run(<[In]()> ByVal tStart As Long) As Integer

        <PreserveSig()> _
        Shadows Function GetState(<[In]()> ByVal dwMilliSecsTimeout As Integer, ByRef State As FilterState) As Integer

        <PreserveSig()> _
        Shadows Function SetSyncSource(<[In](), MarshalAs(UnmanagedType.[Interface])> ByVal pClock As IReferenceClock) As Integer

        <PreserveSig()> _
        Shadows Function GetSyncSource(<MarshalAs(UnmanagedType.[Interface])> ByRef pClock As IReferenceClock) As Integer

        <PreserveSig()> _
        Shadows Function EnumPins(<MarshalAs(UnmanagedType.[Interface])> ByRef ppEnum As IEnumPins) As Integer

        <PreserveSig()> _
        Shadows Function FindPin(<[In](), MarshalAs(UnmanagedType.LPWStr)> ByVal Id As String, <MarshalAs(UnmanagedType.[Interface])> ByRef ppPin As IPin) As Integer

        <PreserveSig()> _
        Shadows Function QueryFilterInfo(ByRef pInfo As FilterInfo) As Integer

        <PreserveSig()> _
        Shadows Function JoinFilterGraph(<[In](), MarshalAs(UnmanagedType.[Interface])> ByVal pGraph As IFilterGraph, <[In](), MarshalAs(UnmanagedType.LPWStr)> ByVal pName As String) As Integer

        <PreserveSig()> _
        Shadows Function QueryVendorInfo(<MarshalAs(UnmanagedType.LPWStr)> ByRef pVendorInfo As String) As Integer

        <PreserveSig()> _
        Function AddMediaStream(<[In](), MarshalAs(UnmanagedType.[Interface])> ByVal pAMMediaStream As IAMMediaStream) As Integer

        <PreserveSig()> _
        Function GetMediaStream(<[In](), MarshalAs(UnmanagedType.LPStruct)> ByVal idPurpose As Guid, <MarshalAs(UnmanagedType.[Interface])> ByRef ppMediaStream As IMediaStream) As Integer

        <PreserveSig()> _
        Function EnumMediaStreams(<[In]()> ByVal Index As Integer, <MarshalAs(UnmanagedType.[Interface])> ByRef ppMediaStream As IMediaStream) As Integer

        <PreserveSig()> _
        Function SupportSeeking(<[In](), MarshalAs(UnmanagedType.Bool)> ByVal bRenderer As Boolean) As Integer

        <PreserveSig()> _
        Function ReferenceTimeToStreamTime(<[In](), Out()> ByRef pTime As Long) As Integer

        <PreserveSig()> _
        Function GetCurrentStreamTime(ByRef pCurrentStreamTime As Long) As Integer

        <PreserveSig()> _
        Function WaitUntil(<[In]()> ByVal WaitStreamTime As Long) As Integer

        <PreserveSig()> _
        Function Flush(<[In](), MarshalAs(UnmanagedType.Bool)> ByVal bCancelEOS As Boolean) As Integer

        <PreserveSig()> _
        Function EndOfStream() As Integer

    End Interface

End Namespace
