
Imports System.Runtime.InteropServices
Imports System.Text

Namespace SBE

    Public Enum RecordingType
        Content = 0 '  no post-recording or overlapped
        Reference   '  allows post-recording & overlapped
    End Enum

    Public Enum StreamBufferAttrDataType
        DWord = 0
        [String] = 1
        Binary = 2
        Bool = 3
        QWord = 4
        Word = 5
        Guid = 6
    End Enum

    Public Enum StreamBufferEventCode
        TimeHole = &H326
        ' STREAMBUFFER_EC_TIMEHOLE
        StaleDataRead
        ' STREAMBUFFER_EC_STALE_DATA_READ
        StaleFileDeleted
        ' STREAMBUFFER_EC_STALE_FILE_DELETED
        ContentBecomingStale
        ' STREAMBUFFER_EC_CONTENT_BECOMING_STALE
        WriteFailure
        ' STREAMBUFFER_EC_WRITE_FAILURE
        ReadFailure
        ' STREAMBUFFER_EC_READ_FAILURE
        RateChanged
        ' STREAMBUFFER_EC_RATE_CHANGED
        PrimaryAudio
        ' STREAMBUFFER_EC_PRIMARY_AUDIO
        ' In Windows 7, Microsoft *RENUMBERED* the elements of the enum.  So, if
        ' you are using Vista, 0x32b is STREAMBUFFER_EC_READ_FAILURE.  In W7, it
        ' is STREAMBUFFER_EC_WRITE_FAILURE_CLEAR.  This also affects RateChanged,
        ' and PrimaryAudio.  If you are using a "case" statement to process events,
        ' you probably need entirely separate case statements for Vista/W7.
        WriteFailureClear_W7 = &H32B
        ' STREAMBUFFER_EC_WRITE_FAILURE_CLEAR
        ReadFailure_W7
        RateChanged_W7
        PrimaryAudio_W7
        RateChangingForSetPositions
        ' STREAMBUFFER_EC_RATE_CHANGING_FOR_SETPOSITIONS
        SetPositionsEventsDone
        ' STREAMBUFFER_EC_SETPOSITIONS_EVENTS_DONE
    End Enum

    Public NotInheritable Class StreamBufferRecording
        Private Sub New()
        End Sub

        Public ReadOnly Duration As String = "Duration"
        Public ReadOnly Bitrate As String = "Bitrate"
        Public ReadOnly Seekable As String = "Seekable"
        Public ReadOnly Stridable As String = "Stridable"
        Public ReadOnly Broadcast As String = "Broadcast"
        Public ReadOnly [Protected] As String = "Is_Protected"
        Public ReadOnly Trusted As String = "Is_Trusted"
        Public ReadOnly Signature_Name As String = "Signature_Name"
        Public ReadOnly HasAudio As String = "HasAudio"
        Public ReadOnly HasImage As String = "HasImage"
        Public ReadOnly HasScript As String = "HasScript"
        Public ReadOnly HasVideo As String = "HasVideo"
        Public ReadOnly CurrentBitrate As String = "CurrentBitrate"
        Public ReadOnly OptimalBitrate As String = "OptimalBitrate"
        Public ReadOnly HasAttachedImages As String = "HasAttachedImages"
        Public ReadOnly SkipBackward As String = "Can_Skip_Backward"
        Public ReadOnly SkipForward As String = "Can_Skip_Forward"
        Public ReadOnly NumberOfFrames As String = "NumberOfFrames"
        Public ReadOnly FileSize As String = "FileSize"
        Public ReadOnly HasArbitraryDataStream As String = "HasArbitraryDataStream"
        Public ReadOnly HasFileTransferStream As String = "HasFileTransferStream"

        ' The content description object supports 5 basic attributes.
        Public ReadOnly Title As String = "Title"
        Public ReadOnly Author As String = "Author"
        Public ReadOnly Description As String = "Description"
        Public ReadOnly Rating As String = "Rating"
        Public ReadOnly Copyright As String = "Copyright"

        ' These attributes are used to configure DRM using IWMDRMWriter::SetDRMAttribute.
        Public ReadOnly Use_DRM As String = "Use_DRM"
        Public ReadOnly DRM_Flags As String = "DRM_Flags"
        Public ReadOnly DRM_Level As String = "DRM_Level"

        ' These are the additional attributes defined in the WM attribute
        ' namespace that give information about the content.
        Public ReadOnly AlbumTitle As String = "WM/AlbumTitle"

        Public ReadOnly Track As String = "WM/Track"
        Public ReadOnly PromotionURL As String = "WM/PromotionURL"
        Public ReadOnly AlbumCoverURL As String = "WM/AlbumCoverURL"
        Public ReadOnly Genre As String = "WM/Genre"
        Public ReadOnly Year As String = "WM/Year"
        Public ReadOnly GenreID As String = "WM/GenreID"
        Public ReadOnly MCDI As String = "WM/MCDI"
        Public ReadOnly Composer As String = "WM/Composer"
        Public ReadOnly Lyrics As String = "WM/Lyrics"
        Public ReadOnly TrackNumber As String = "WM/TrackNumber"
        Public ReadOnly ToolName As String = "WM/ToolName"
        Public ReadOnly ToolVersion As String = "WM/ToolVersion"
        Public ReadOnly IsVBR As String = "IsVBR"
        Public ReadOnly AlbumArtist As String = "WM/AlbumArtist"

        ' These optional attributes may be used to give information
        ' about the branding of the content.
        Public ReadOnly BannerImageType As String = "BannerImageType"
        Public ReadOnly BannerImageData As String = "BannerImageData"
        Public ReadOnly BannerImageURL As String = "BannerImageURL"
        Public ReadOnly CopyrightURL As String = "CopyrightURL"

        ' Optional attributes, used to give information
        ' about video stream properties.
        Public ReadOnly AspectRatioX As String = "AspectRatioX"
        Public ReadOnly AspectRatioY As String = "AspectRatioY"

        ' The NSC file supports the following attributes.
        Public ReadOnly NSCName As String = "NSC_Name"
        Public ReadOnly NSCAddress As String = "NSC_Address"
        Public ReadOnly NSCPhone As String = "NSC_Phone"
        Public ReadOnly NSCEmail As String = "NSC_Email"
        Public ReadOnly NSCDescription As String = "NSC_Description"
    End Class

    <StructLayout(LayoutKind.Sequential)> _
    Public Structure StreamBufferAttribute
        <MarshalAs(UnmanagedType.LPWStr)> _
        Public pszName As String
        Public StreamBufferAttributeType As StreamBufferAttrDataType
        Public pbAttribute As IntPtr  ' BYTE *
        Public cbLength As Short
    End Structure

    <StructLayout(LayoutKind.Sequential)> _
    Public Structure SBEPinData
        Public cDataBytes As Long           '  total sample payload bytes
        Public cSamplesProcessed As Long    '  samples processed
        Public cDiscontinuities As Long     '  number of discontinuities
        Public cSyncPoints As Long          '  number of syncpoints
        Public cTimestamps As Long          '  number of timestamps
    End Structure

    Public NotInheritable Class SBEEvent
        Public Shared ReadOnly RecControlStarted As New Guid(&H8966A89EUI, &HF83E, &H4C0E, &HBC, &H3B, &HBF, _
         &HA7, &H64, &H9E, &H4, &HCB)

        Public Shared ReadOnly RecControlStopped As New Guid(&H454B1EC8, &HC9B, &H4CAA, &HB1, &HA1, &H1E, _
         &H7A, &H26, &H66, &HF6, &HC3)

        Public Shared ReadOnly StreamDescEvent As New Guid(&H2313A4ED, &HBF2D, &H454F, &HAD, &H8A, &HD9, _
         &H5B, &HA7, &HF9, &H1F, &HEE)

        Public Shared ReadOnly V1StreamsCreationEvent As New Guid(&HFCF09, &H97F5, &H46AC, &H97, &H69, &H7A, _
         &H83, &HB3, &H53, &H84, &HFB)

        Public Shared ReadOnly V2StreamsCreationEvent As New Guid(&HA72530A3UI, &H344, &H4CAB, &HA2, &HD0, &HFE, _
         &H93, &H7D, &HBD, &HCA, &HB3)
    End Class

    <Flags()> _
    Public Enum CrossbarDefaultFlags
        None = &H0
        Profile = &H1
        Streams = &H2
    End Enum

    <StructLayout(LayoutKind.Sequential)> _
    Public Class SBE2_StreamDesc
        Public Version As Integer
        Public StreamId As Integer
        Public [Default] As Integer
        Public Reserved As Integer
    End Class

    <StructLayout(LayoutKind.Sequential)> _
    Public Class DVRStreamDesc
        Public Version As Integer
        Public StreamId As Integer
        Public [Default] As Boolean
        Public Creation As Boolean
        Public Reserved As Integer
        Public guidSubMediaType As Guid
        Public guidFormatType As Guid
        Public MediaType As AMMediaType
    End Class


    <ComImport(), System.Security.SuppressUnmanagedCodeSecurity(), Guid("7E2D2A1E-7192-4bd7-80C1-061FD1D10402"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)> _
    Public Interface IStreamBufferConfigure3
        Inherits IStreamBufferConfigure2

        <PreserveSig()> _
        Shadows Function SetDirectory(<[In](), MarshalAs(UnmanagedType.LPWStr)> ByVal pszDirectoryName As String) As Integer

        <PreserveSig()> _
        Shadows Function GetDirectory(<Out(), MarshalAs(UnmanagedType.LPWStr)> ByRef pszDirectoryName As String) As Integer

        <PreserveSig()> _
        Shadows Function SetBackingFileCount(<[In]()> ByVal dwMin As Integer, <[In]()> ByVal dwMax As Integer) As Integer

        <PreserveSig()> _
        Shadows Function GetBackingFileCount(<Out()> ByRef dwMin As Integer, <Out()> ByRef dwMax As Integer) As Integer

        <PreserveSig()> _
        Shadows Function SetBackingFileDuration(<[In]()> ByVal dwSeconds As Integer) As Integer

        <PreserveSig()> _
        Shadows Function GetBackingFileDuration(<Out()> ByRef pdwSeconds As Integer) As Integer

        <PreserveSig()> _
        Shadows Function SetMultiplexedPacketSize(<[In]()> ByVal cbBytesPerPacket As Integer) As Integer

        <PreserveSig()> _
        Shadows Function GetMultiplexedPacketSize(<Out()> ByRef pcbBytesPerPacket As Integer) As Integer

        <PreserveSig()> _
        Shadows Function SetFFTransitionRates(<[In]()> ByVal dwMaxFullFrameRate As Integer, <[In]()> ByVal dwMaxNonSkippingRate As Integer) As Integer

        <PreserveSig()> _
        Shadows Function GetFFTransitionRates(<Out()> ByRef pdwMaxFullFrameRate As Integer, <Out()> ByRef pdwMaxNonSkippingRate As Integer) As Integer

        <PreserveSig()> _
        Function SetStartRecConfig(<[In](), MarshalAs(UnmanagedType.Bool)> ByVal fStartStopsCur As Boolean) As Integer

        <PreserveSig()> _
        Function GetStartRecConfig(<Out(), MarshalAs(UnmanagedType.Bool)> ByRef pfStartStopsCur As Boolean) As Integer

        <PreserveSig()> _
        Function SetNamespace(<[In](), MarshalAs(UnmanagedType.LPWStr)> ByVal pszNamespace As String) As Integer

        <PreserveSig()> _
        Function GetNamespace(<Out(), MarshalAs(UnmanagedType.LPWStr)> ByRef ppszNamespace As String) As Integer
    End Interface

    <ComImport(), System.Security.SuppressUnmanagedCodeSecurity(), Guid("9ce50f2d-6ba7-40fb-a034-50b1a674ec78"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)> _
    Public Interface IStreamBufferInitialize
        <PreserveSig()> _
        Function SetHKEY(<[In]()> ByVal hkeyRoot As IntPtr) As Integer
        ' HKEY
        ' PSID *
        <PreserveSig()> _
        Function SetSIDs(<[In]()> ByVal cSIDs As Integer, <[In](), MarshalAs(UnmanagedType.LPArray)> ByVal ppSID As IntPtr()) As Integer
    End Interface

    <ComImport(), System.Security.SuppressUnmanagedCodeSecurity(), Guid("afd1f242-7efd-45ee-ba4e-407a25c9a77a"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)> _
    Public Interface IStreamBufferSink
        <PreserveSig()> _
        Function LockProfile(<[In](), MarshalAs(UnmanagedType.LPWStr)> ByVal pszStreamBufferFilename As String) As Integer

        <PreserveSig()> _
        Function CreateRecorder(<[In](), MarshalAs(UnmanagedType.LPWStr)> ByVal pszFilename As String, <[In]()> ByVal dwRecordType As RecordingType, <Out(), MarshalAs(UnmanagedType.IUnknown)> ByRef pRecordingIUnknown As Object) As Integer

        <PreserveSig()> _
        Function IsProfileLocked() As Integer
    End Interface

    <ComImport(), System.Security.SuppressUnmanagedCodeSecurity(), Guid("DB94A660-F4FB-4bfa-BCC6-FE159A4EEA93"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)> _
    Public Interface IStreamBufferSink2
        Inherits IStreamBufferSink

        <PreserveSig()> _
        Shadows Function LockProfile(<[In](), MarshalAs(UnmanagedType.LPWStr)> ByVal pszStreamBufferFilename As String) As Integer

        <PreserveSig()> _
        Shadows Function CreateRecorder(<[In](), MarshalAs(UnmanagedType.LPWStr)> ByVal pszFilename As String, <[In]()> ByVal dwRecordType As RecordingType, <Out(), MarshalAs(UnmanagedType.IUnknown)> ByRef pRecordingIUnknown As Object) As Integer

        <PreserveSig()> _
        Shadows Function IsProfileLocked() As Integer

        <PreserveSig()> _
        Function UnlockProfile() As Integer
    End Interface

    <ComImport(), System.Security.SuppressUnmanagedCodeSecurity(), Guid("974723f2-887a-4452-9366-2cff3057bc8f"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)> _
    Public Interface IStreamBufferSink3
        Inherits IStreamBufferSink2

        <PreserveSig()> _
        Shadows Function LockProfile(<[In](), MarshalAs(UnmanagedType.LPWStr)> ByVal pszStreamBufferFilename As String) As Integer

        <PreserveSig()> _
        Shadows Function CreateRecorder(<[In](), MarshalAs(UnmanagedType.LPWStr)> ByVal pszFilename As String, <[In]()> ByVal dwRecordType As RecordingType, <Out(), MarshalAs(UnmanagedType.IUnknown)> ByRef pRecordingIUnknown As Object) As Integer

        <PreserveSig()> _
        Shadows Function IsProfileLocked() As Integer

        <PreserveSig()> _
        Shadows Function UnlockProfile() As Integer

        <PreserveSig()> _
        Function SetAvailableFilter(<[In](), Out()> ByRef prtMin As Long) As Integer
    End Interface

    <ComImport(), System.Security.SuppressUnmanagedCodeSecurity(), Guid("1c5bd776-6ced-4f44-8164-5eab0e98db12"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)> _
    Public Interface IStreamBufferSource
        <PreserveSig()> _
        Function SetStreamSink(<[In]()> ByVal pIStreamBufferSink As IStreamBufferSink) As Integer
    End Interface

    <ComImport(), System.Security.SuppressUnmanagedCodeSecurity(), Guid("ba9b6c99-f3c7-4ff2-92db-cfdd4851bf31"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)> _
    Public Interface IStreamBufferRecordControl
        <PreserveSig()> _
        Function Start(<[In](), Out()> ByRef prtStart As Long) As Integer

        <PreserveSig()> _
        Function [Stop](<[In]()> ByVal rtStop As Long) As Integer

        <PreserveSig()> _
        Function GetRecordingStatus(<Out()> ByRef phResult As Integer, <Out(), MarshalAs(UnmanagedType.Bool)> ByRef pbStarted As Boolean, <Out(), MarshalAs(UnmanagedType.Bool)> ByRef pbStopped As Boolean) As Integer
    End Interface

    <ComImport(), System.Security.SuppressUnmanagedCodeSecurity(), Guid("9E259A9B-8815-42ae-B09F-221970B154FD"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)> _
    Public Interface IStreamBufferRecComp
        <PreserveSig()> _
        Function Initialize(<[In](), MarshalAs(UnmanagedType.LPWStr)> ByVal pszTargetFilename As String, <[In](), MarshalAs(UnmanagedType.LPWStr)> ByVal pszSBRecProfileRef As String) As Integer

        <PreserveSig()> _
        Function Append(<[In](), MarshalAs(UnmanagedType.LPWStr)> ByVal pszSBRecording As String) As Integer

        <PreserveSig()> _
        Function AppendEx(<[In](), MarshalAs(UnmanagedType.LPWStr)> ByVal pszSBRecording As String, <[In]()> ByVal rtStart As Long, <[In]()> ByVal rtStop As Long) As Integer

        <PreserveSig()> _
        Function GetCurrentLength(<Out()> ByRef pcSeconds As Integer) As Integer

        <PreserveSig()> _
        Function Close() As Integer

        <PreserveSig()> _
        Function Cancel() As Integer
    End Interface

    <ComImport(), System.Security.SuppressUnmanagedCodeSecurity(), Guid("16CA4E03-FE69-4705-BD41-5B7DFC0C95F3"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)> _
    Public Interface IStreamBufferRecordingAttribute
        ' BYTE *
        <PreserveSig()> _
        Function SetAttribute(<[In]()> ByVal ulReserved As Integer, <[In](), MarshalAs(UnmanagedType.LPWStr)> ByVal pszAttributeName As String, <[In]()> ByVal StreamBufferAttributeType As StreamBufferAttrDataType, <[In]()> ByVal pbAttribute As IntPtr, <[In]()> ByVal cbAttributeLength As Short) As Integer

        <PreserveSig()> _
        Function GetAttributeCount(<[In]()> ByVal ulReserved As Integer, <Out()> ByRef pcAttributes As Short) As Integer

        ' BYTE *
        <PreserveSig()> _
        Function GetAttributeByName(<[In](), MarshalAs(UnmanagedType.LPWStr)> ByVal pszAttributeName As String, <[In]()> ByVal pulReserved As Integer, <Out()> ByRef pStreamBufferAttributeType As StreamBufferAttrDataType, <[In](), Out()> ByVal pbAttribute As IntPtr, <[In](), Out()> ByRef pcbLength As Short) As Integer

        ' BYTE *
        <PreserveSig()> _
        Function GetAttributeByIndex(<[In]()> ByVal wIndex As Short, <[In]()> ByVal pulReserved As Integer, <Out(), MarshalAs(UnmanagedType.LPWStr)> ByVal pszAttributeName As StringBuilder, <[In](), Out()> ByRef pcchNameLength As Short, <Out()> ByRef pStreamBufferAttributeType As StreamBufferAttrDataType, ByVal pbAttribute As IntPtr, _
         <[In](), Out()> ByRef pcbLength As Short) As Integer

        Function EnumAttributes(<Out()> ByRef ppIEnumStreamBufferAttrib As IEnumStreamBufferRecordingAttrib) As Integer
    End Interface

    <ComImport(), System.Security.SuppressUnmanagedCodeSecurity(), Guid("C18A9162-1E82-4142-8C73-5690FA62FE33"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)> _
    Public Interface IEnumStreamBufferRecordingAttrib
        <PreserveSig()> _
        Function [Next](<[In]()> ByVal cRequest As Integer, <Out(), MarshalAs(UnmanagedType.LPArray, SizeParamIndex:=0)> ByVal pStreamBufferAttribute As StreamBufferAttribute(), <[In]()> ByVal pcReceived As IntPtr) As Integer

        <PreserveSig()> _
        Function Skip(<[In]()> ByVal cRecords As Integer) As Integer

        <PreserveSig()> _
        Function Reset() As Integer

        <PreserveSig()> _
        Function Clone(<Out()> ByRef ppIEnumStreamBufferAttrib As IEnumStreamBufferRecordingAttrib) As Integer
    End Interface

    <ComImport(), System.Security.SuppressUnmanagedCodeSecurity(), Guid("ce14dfae-4098-4af7-bbf7-d6511f835414"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)> _
    Public Interface IStreamBufferConfigure
        <PreserveSig()> _
        Function SetDirectory(<[In](), MarshalAs(UnmanagedType.LPWStr)> ByVal pszDirectoryName As String) As Integer

        <PreserveSig()> _
        Function GetDirectory(<Out(), MarshalAs(UnmanagedType.LPWStr)> ByRef pszDirectoryName As String) As Integer

        <PreserveSig()> _
        Function SetBackingFileCount(<[In]()> ByVal dwMin As Integer, <[In]()> ByVal dwMax As Integer) As Integer

        <PreserveSig()> _
        Function GetBackingFileCount(<Out()> ByRef dwMin As Integer, <Out()> ByRef dwMax As Integer) As Integer

        <PreserveSig()> _
        Function SetBackingFileDuration(<[In]()> ByVal dwSeconds As Integer) As Integer

        <PreserveSig()> _
        Function GetBackingFileDuration(<Out()> ByRef pdwSeconds As Integer) As Integer
    End Interface

    <ComImport(), System.Security.SuppressUnmanagedCodeSecurity(), Guid("53E037BF-3992-4282-AE34-2487B4DAE06B"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)> _
    Public Interface IStreamBufferConfigure2
        Inherits IStreamBufferConfigure

        <PreserveSig()> _
        Shadows Function SetDirectory(<[In](), MarshalAs(UnmanagedType.LPWStr)> ByVal pszDirectoryName As String) As Integer

        <PreserveSig()> _
        Shadows Function GetDirectory(<Out(), MarshalAs(UnmanagedType.LPWStr)> ByRef pszDirectoryName As String) As Integer

        <PreserveSig()> _
        Shadows Function SetBackingFileCount(<[In]()> ByVal dwMin As Integer, <[In]()> ByVal dwMax As Integer) As Integer

        <PreserveSig()> _
        Shadows Function GetBackingFileCount(<Out()> ByRef dwMin As Integer, <Out()> ByRef dwMax As Integer) As Integer

        <PreserveSig()> _
        Shadows Function SetBackingFileDuration(<[In]()> ByVal dwSeconds As Integer) As Integer

        <PreserveSig()> _
        Shadows Function GetBackingFileDuration(<Out()> ByRef pdwSeconds As Integer) As Integer

        <PreserveSig()> _
        Function SetMultiplexedPacketSize(<[In]()> ByVal cbBytesPerPacket As Integer) As Integer

        <PreserveSig()> _
        Function GetMultiplexedPacketSize(<Out()> ByRef pcbBytesPerPacket As Integer) As Integer

        <PreserveSig()> _
        Function SetFFTransitionRates(<[In]()> ByVal dwMaxFullFrameRate As Integer, <[In]()> ByVal dwMaxNonSkippingRate As Integer) As Integer

        <PreserveSig()> _
        Function GetFFTransitionRates(<Out()> ByRef pdwMaxFullFrameRate As Integer, <Out()> ByRef pdwMaxNonSkippingRate As Integer) As Integer
    End Interface

    <ComImport(), System.Security.SuppressUnmanagedCodeSecurity(), Guid("f61f5c26-863d-4afa-b0ba-2f81dc978596"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)> _
    Public Interface IStreamBufferMediaSeeking
        Inherits IMediaSeeking

        <PreserveSig()> _
        Shadows Function GetCapabilities(<Out()> ByRef pCapabilities As AMSeekingSeekingCapabilities) As Integer

        <PreserveSig()> _
        Shadows Function CheckCapabilities(<[In](), Out()> ByRef pCapabilities As AMSeekingSeekingCapabilities) As Integer

        <PreserveSig()> _
        Shadows Function IsFormatSupported(<[In](), MarshalAs(UnmanagedType.LPStruct)> ByVal pFormat As Guid) As Integer

        <PreserveSig()> _
        Shadows Function QueryPreferredFormat(<Out()> ByRef pFormat As Guid) As Integer

        <PreserveSig()> _
        Shadows Function GetTimeFormat(<Out()> ByRef pFormat As Guid) As Integer

        <PreserveSig()> _
        Shadows Function IsUsingTimeFormat(<[In](), MarshalAs(UnmanagedType.LPStruct)> ByVal pFormat As Guid) As Integer

        <PreserveSig()> _
        Shadows Function SetTimeFormat(<[In](), MarshalAs(UnmanagedType.LPStruct)> ByVal pFormat As Guid) As Integer

        <PreserveSig()> _
        Shadows Function GetDuration(<Out()> ByRef pDuration As Long) As Integer

        <PreserveSig()> _
        Shadows Function GetStopPosition(<Out()> ByRef pStop As Long) As Integer

        <PreserveSig()> _
        Shadows Function GetCurrentPosition(<Out()> ByRef pCurrent As Long) As Integer

        <PreserveSig()> _
        Shadows Function ConvertTimeFormat(<Out()> ByRef pTarget As Long, <[In](), MarshalAs(UnmanagedType.LPStruct)> ByVal pTargetFormat As DsGuid, <[In]()> ByVal Source As Long, <[In](), MarshalAs(UnmanagedType.LPStruct)> ByVal pSourceFormat As DsGuid) As Integer

        <PreserveSig()> _
        Shadows Function SetPositions(<[In](), Out(), MarshalAs(UnmanagedType.LPStruct)> ByVal pCurrent As DsLong, <[In]()> ByVal dwCurrentFlags As AMSeekingSeekingFlags, <[In](), Out(), MarshalAs(UnmanagedType.LPStruct)> ByVal pStop As DsLong, <[In]()> ByVal dwStopFlags As AMSeekingSeekingFlags) As Integer

        <PreserveSig()> _
        Shadows Function GetPositions(<Out()> ByRef pCurrent As Long, <Out()> ByRef pStop As Long) As Integer

        <PreserveSig()> _
        Shadows Function GetAvailable(<Out()> ByRef pEarliest As Long, <Out()> ByRef pLatest As Long) As Integer

        <PreserveSig()> _
        Shadows Function SetRate(<[In]()> ByVal dRate As Double) As Integer

        <PreserveSig()> _
        Shadows Function GetRate(<Out()> ByRef pdRate As Double) As Integer

        <PreserveSig()> _
        Shadows Function GetPreroll(<Out()> ByRef pllPreroll As Long) As Integer

    End Interface

    <ComImport(), System.Security.SuppressUnmanagedCodeSecurity(), Guid("3a439ab0-155f-470a-86a6-9ea54afd6eaf"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)> _
    Public Interface IStreamBufferMediaSeeking2
        Inherits IStreamBufferMediaSeeking

        <PreserveSig()> _
        Shadows Function GetCapabilities(<Out()> ByRef pCapabilities As AMSeekingSeekingCapabilities) As Integer

        <PreserveSig()> _
        Shadows Function CheckCapabilities(<[In](), Out()> ByRef pCapabilities As AMSeekingSeekingCapabilities) As Integer

        <PreserveSig()> _
        Shadows Function IsFormatSupported(<[In](), MarshalAs(UnmanagedType.LPStruct)> ByVal pFormat As Guid) As Integer

        <PreserveSig()> _
        Shadows Function QueryPreferredFormat(<Out()> ByRef pFormat As Guid) As Integer

        <PreserveSig()> _
        Shadows Function GetTimeFormat(<Out()> ByRef pFormat As Guid) As Integer

        <PreserveSig()> _
        Shadows Function IsUsingTimeFormat(<[In](), MarshalAs(UnmanagedType.LPStruct)> ByVal pFormat As Guid) As Integer

        <PreserveSig()> _
        Shadows Function SetTimeFormat(<[In](), MarshalAs(UnmanagedType.LPStruct)> ByVal pFormat As Guid) As Integer

        <PreserveSig()> _
        Shadows Function GetDuration(<Out()> ByRef pDuration As Long) As Integer

        <PreserveSig()> _
        Shadows Function GetStopPosition(<Out()> ByRef pStop As Long) As Integer

        <PreserveSig()> _
        Shadows Function GetCurrentPosition(<Out()> ByRef pCurrent As Long) As Integer

        <PreserveSig()> _
        Shadows Function ConvertTimeFormat(<Out()> ByRef pTarget As Long, <[In](), MarshalAs(UnmanagedType.LPStruct)> ByVal pTargetFormat As DsGuid, <[In]()> ByVal Source As Long, <[In](), MarshalAs(UnmanagedType.LPStruct)> ByVal pSourceFormat As DsGuid) As Integer

        <PreserveSig()> _
        Shadows Function SetPositions(<[In](), Out(), MarshalAs(UnmanagedType.LPStruct)> ByVal pCurrent As DsLong, <[In]()> ByVal dwCurrentFlags As AMSeekingSeekingFlags, <[In](), Out(), MarshalAs(UnmanagedType.LPStruct)> ByVal pStop As DsLong, <[In]()> ByVal dwStopFlags As AMSeekingSeekingFlags) As Integer

        <PreserveSig()> _
        Shadows Function GetPositions(<Out()> ByRef pCurrent As Long, <Out()> ByRef pStop As Long) As Integer

        <PreserveSig()> _
        Shadows Function GetAvailable(<Out()> ByRef pEarliest As Long, <Out()> ByRef pLatest As Long) As Integer

        <PreserveSig()> _
        Shadows Function SetRate(<[In]()> ByVal dRate As Double) As Integer

        <PreserveSig()> _
        Shadows Function GetRate(<Out()> ByRef pdRate As Double) As Integer

        <PreserveSig()> _
        Shadows Function GetPreroll(<Out()> ByRef pllPreroll As Long) As Integer

        <PreserveSig()> _
        Function SetRateEx(<[In]()> ByVal dRate As Double, <[In]()> ByVal dwFramesPerSec As Integer) As Integer
    End Interface

    <ComImport(), System.Security.SuppressUnmanagedCodeSecurity(), Guid("9D2A2563-31AB-402e-9A6B-ADB903489440"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)> _
    Public Interface IStreamBufferDataCounters
        <PreserveSig()> _
        Function GetData(<Out()> ByRef pPinData As SBEPinData) As Integer

        <PreserveSig()> _
        Function ResetData() As Integer
    End Interface

    <ComImport(), System.Security.SuppressUnmanagedCodeSecurity(), Guid("caede759-b6b1-11db-a578-0018f3fa24c6"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)> _
    Public Interface ISBE2GlobalEvent
        <PreserveSig()> _
        Function GetEvent(<[In](), MarshalAs(UnmanagedType.LPStruct)> ByVal idEvt As Guid, ByVal param1 As Integer, ByVal param2 As Integer, ByVal param3 As Integer, ByVal param4 As Integer, <MarshalAs(UnmanagedType.Bool)> ByRef pSpanning As Boolean, _
         ByRef pcb As Integer, <Out()> ByVal pb As IntPtr) As Integer
    End Interface

    <ComImport(), System.Security.SuppressUnmanagedCodeSecurity(), Guid("6D8309BF-00FE-4506-8B03-F8C65B5C9B39"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)> _
    Public Interface ISBE2GlobalEvent2
        Inherits ISBE2GlobalEvent

        <PreserveSig()> _
        Shadows Function GetEvent(<[In](), MarshalAs(UnmanagedType.LPStruct)> ByVal idEvt As Guid, ByVal param1 As Integer, ByVal param2 As Integer, ByVal param3 As Integer, ByVal param4 As Integer, <Out(), MarshalAs(UnmanagedType.Bool)> ByRef pSpanning As Boolean, _
         ByRef pcb As Integer, <Out()> ByVal pb As IntPtr) As Integer

        <PreserveSig()> _
        Function GetEventEx(<[In](), MarshalAs(UnmanagedType.LPStruct)> ByVal idEvt As Guid, ByVal param1 As Integer, ByVal param2 As Integer, ByVal param3 As Integer, ByVal param4 As Integer, <MarshalAs(UnmanagedType.Bool)> ByRef pSpanning As Boolean, _
         ByRef pcb As Integer, <Out()> ByVal pb As IntPtr, ByRef pStreamTime As Long) As Integer
    End Interface

    <ComImport(), System.Security.SuppressUnmanagedCodeSecurity(), Guid("caede760-b6b1-11db-a578-0018f3fa24c6"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)> _
    Public Interface ISBE2SpanningEvent
        <PreserveSig()> _
        Function GetEvent(<[In](), MarshalAs(UnmanagedType.LPStruct)> ByVal idEvt As Guid, ByVal streamId As Integer, ByRef pcb As Integer, ByVal pb As IntPtr) As Integer
    End Interface

    <ComImport(), System.Security.SuppressUnmanagedCodeSecurity(), Guid("547b6d26-3226-487e-8253-8aa168749434"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)> _
    Public Interface ISBE2Crossbar
        <PreserveSig()> _
        Function EnableDefaultMode(ByVal DefaultFlags As CrossbarDefaultFlags) As Integer

        <PreserveSig()> _
        Function GetInitialProfile(ByRef ppProfile As ISBE2MediaTypeProfile) As Integer

        <PreserveSig()> _
        Function SetOutputProfile(ByVal pProfile As ISBE2MediaTypeProfile, ByRef pcOutputPins As Integer, <Out(), MarshalAs(UnmanagedType.LPArray)> ByVal ppOutputPins As IPin()) As Integer

        <PreserveSig()> _
        Function EnumStreams(ByRef ppStreams As ISBE2EnumStream) As Integer
    End Interface

    <ComImport(), System.Security.SuppressUnmanagedCodeSecurity(), Guid("667c7745-85b1-4c55-ae55-4e25056159fc"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)> _
    Public Interface ISBE2StreamMap
        <PreserveSig()> _
        Function MapStream(ByVal Stream As Integer) As Integer

        <PreserveSig()> _
        Function UnmapStream(ByVal Stream As Integer) As Integer

        <PreserveSig()> _
        Function EnumMappedStreams(ByRef ppStreams As ISBE2EnumStream) As Integer
    End Interface

    <ComImport(), System.Security.SuppressUnmanagedCodeSecurity(), Guid("f7611092-9fbc-46ec-a7c7-548ea78b71a4"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)> _
    Public Interface ISBE2EnumStream
        <PreserveSig()> _
        Function [Next](ByVal cRequest As Integer, <[In](), Out(), MarshalAs(UnmanagedType.CustomMarshaler, MarshalTypeRef:=GetType(SDMarshaler))> ByVal pStreamDesc As SBE2_StreamDesc(), ByVal pcReceived As IntPtr) As Integer

        <PreserveSig()> _
        Function Skip(ByVal cRecords As Integer) As Integer

        <PreserveSig()> _
        Function Reset() As Integer

        <PreserveSig()> _
        Function Clone(ByRef ppIEnumStream As ISBE2EnumStream) As Integer
    End Interface

    <ComImport(), System.Security.SuppressUnmanagedCodeSecurity(), Guid("f238267d-4671-40d7-997e-25dc32cfed2a"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)> _
    Public Interface ISBE2MediaTypeProfile
        <PreserveSig()> _
        Function GetStreamCount(ByRef pCount As Integer) As Integer

        <PreserveSig()> _
        Function GetStream(ByVal Index As Integer, ByRef ppMediaType As AMMediaType) As Integer

        <PreserveSig()> _
        Function AddStream(<[In](), MarshalAs(UnmanagedType.LPStruct)> ByVal pMediaType As AMMediaType) As Integer

        <PreserveSig()> _
        Function DeleteStream(ByVal Index As Integer) As Integer
    End Interface

    <ComImport(), System.Security.SuppressUnmanagedCodeSecurity(), Guid("3E2BF5A5-4F96-4899-A1A3-75E8BE9A5AC0"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)> _
    Public Interface ISBE2FileScan
        <PreserveSig()> _
        Function RepairFile(<[In](), MarshalAs(UnmanagedType.LPWStr)> ByVal filename As String) As Integer
    End Interface


    Friend Class SDMarshaler
        Implements ICustomMarshaler
        Private m_sd As SBE2_StreamDesc()

        Public Overridable Sub CleanUpManagedData(ByVal managedObj As Object) Implements ICustomMarshaler.CleanUpManagedData
            m_sd = Nothing
        End Sub

        Public Sub CleanUpNativeData(ByVal pNativeData As IntPtr) Implements ICustomMarshaler.CleanUpNativeData
            If pNativeData <> IntPtr.Zero Then
                Marshal.FreeCoTaskMem(pNativeData)
            End If
        End Sub

        Public Function GetNativeDataSize() As Integer Implements ICustomMarshaler.GetNativeDataSize
            Return 0
        End Function

        Public Function MarshalManagedToNative(ByVal managedObj As Object) As IntPtr Implements ICustomMarshaler.MarshalManagedToNative
            Dim ip As IntPtr

            m_sd = TryCast(managedObj, SBE2_StreamDesc())

            Dim iSize As Integer = m_sd.Length * Marshal.SizeOf(GetType(SBE2_StreamDesc))

            ip = Marshal.AllocCoTaskMem(iSize)

#If DEBUG Then
            For x As Integer = 0 To iSize \ 8 - 1
                Marshal.WriteInt64(ip, x * 8, 0)
            Next
#End If

            Return ip
        End Function

        ' Called just after invoking the COM method.  The IntPtr is the same one that just got returned
        ' from MarshalManagedToNative.  The return value is unused.
        Public Function MarshalNativeToManaged(ByVal pNativeData As IntPtr) As Object Implements ICustomMarshaler.MarshalNativeToManaged
            Dim ip As IntPtr = pNativeData

            For x As Integer = 0 To m_sd.Length - 1
                'm_sd[x] = new SBE2_StreamDesc(); Marshal.PtrToStructure(ip, m_sd[x]);
                m_sd(x) = DirectCast(Marshal.PtrToStructure(ip, GetType(SBE2_StreamDesc)), SBE2_StreamDesc)

                ip = New IntPtr(ip.ToInt64() + Marshal.SizeOf(GetType(SBE2_StreamDesc)))
            Next

            Return Nothing
        End Function

        ' This method is called by interop to create the custom marshaler.  The (optional)
        ' cookie is the value specified in MarshalCookie="asdf", or "" is none is specified.
        Public Shared Function GetInstance(ByVal cookie As String) As ICustomMarshaler
            Return New SDMarshaler()
        End Function
    End Class


End Namespace
