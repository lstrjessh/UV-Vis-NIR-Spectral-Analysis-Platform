
Imports System.Runtime.InteropServices

Namespace MultimediaStreaming

    Public NotInheritable Class MsResults
        ' Prevent people from trying to instantiate this class
        Private Sub New()
        End Sub

        Public Const S_Pending As Integer = &H40001
        Public Const S_NoUpdate As Integer = &H40002
        Public Const S_EndOfStream As Integer = &H40003
        Public Const E_SampleAlloc As Integer = &H80040401
        Public Const E_PurposeId As Integer = &H80040402
        Public Const E_NoStream As Integer = &H80040403
        Public Const E_NoSeeking As Integer = &H80040404
        Public Const E_Incompatible As Integer = &H80040405
        Public Const E_Busy As Integer = &H80040406
        Public Const E_NotInit As Integer = &H80040407
        Public Const E_SourceAlreadyDefined As Integer = &H80040408
        Public Const E_InvalidStreamType As Integer = &H80040409
        Public Const E_NotRunning As Integer = &H8004040A
    End Class

    Public NotInheritable Class MsError
        ' Prevent people from trying to instantiate this class
        Private Sub New()
        End Sub

        Public Shared Function GetErrorText(ByVal hr As Integer) As String
            Dim sRet As String = Nothing

            Select Case hr
                Case MsResults.S_Pending
                    sRet = "Sample update is not yet complete."
                    Exit Select
                Case MsResults.S_NoUpdate
                    sRet = "Sample was not updated after forced completion."
                    Exit Select
                Case MsResults.S_EndOfStream
                    sRet = "End of stream. Sample not updated."
                    Exit Select
                Case MsResults.E_SampleAlloc
                    sRet = "An IMediaStream object could not be removed from an IMultiMediaStream object because it still contains at least one allocated sample."
                    Exit Select
                Case MsResults.E_PurposeId
                    sRet = "The specified purpose ID can't be used for the call."
                    Exit Select
                Case MsResults.E_NoStream
                    sRet = "No stream can be found with the specified attributes."
                    Exit Select
                Case MsResults.E_NoSeeking
                    sRet = "Seeking not supported for this IMultiMediaStream object."
                    Exit Select
                Case MsResults.E_Incompatible
                    sRet = "The stream formats are not compatible."
                    Exit Select
                Case MsResults.E_Busy
                    sRet = "The sample is busy."
                    Exit Select
                Case MsResults.E_NotInit
                    sRet = "The object can't accept the call because its initialize function or equivalent has not been called."
                    Exit Select
                Case MsResults.E_SourceAlreadyDefined
                    sRet = "Source already defined."
                    Exit Select
                Case MsResults.E_InvalidStreamType
                    sRet = "The stream type is not valid for this operation."
                    Exit Select
                Case MsResults.E_NotRunning
                    sRet = "The IMultiMediaStream object is not in running state."
                    Exit Select
                Case Else
                    sRet = DsError.GetErrorText(hr)
                    Exit Select
            End Select

            Return sRet
        End Function

        Public Shared Sub ThrowExceptionForHR(ByVal hr As Integer)
            ' If an error has occurred
            If hr < 0 Then
                ' If a string is returned, build a com error from it
                Dim buf As String = GetErrorText(hr)

                If buf IsNot Nothing Then
                    Throw New COMException(buf, hr)
                Else
                    ' No string, just use standard com error
                    Marshal.ThrowExceptionForHR(hr)
                End If
            End If
        End Sub
    End Class


    <ComImport(), Guid("49c47ce5-9ba4-11d0-8212-00c04fc32c45")> _
    Public Class AMMultiMediaStream
    End Class

    <ComImport(), Guid("CF0F2F7C-F7BF-11d0-900D-00C04FD9189D")> _
    Public Class AMMediaTypeStream
    End Class

    <ComImport(), Guid("49c47ce4-9ba4-11d0-8212-00c04fc32c45")> _
    Public Class AMDirectDrawStream
    End Class

    <ComImport(), Guid("8496e040-af4c-11d0-8212-00c04fc32c45")> _
    Public Class AMAudioStream
    End Class

    <ComImport(), Guid("f2468580-af8a-11d0-8212-00c04fc32c45")> _
    Public Class AMAudioData
    End Class


    <Flags()> _
    Public Enum CompletionStatusFlags
        None = &H0
        NoUpdateOk = &H1
        Wait = &H2
        Abort = &H4
    End Enum

    <Flags()> _
    Public Enum SSUpdate
        None = &H0
        ASync = &H1
        Continuous = &H2
    End Enum

    Public Enum StreamState
        ' Fields
        Run = 1
        [Stop] = 0
    End Enum

    Public Enum StreamType
        ' Fields
        Read = 0
        Transform = 2
        Write = 1
    End Enum

    Public Enum MMSSF
        HasClock = &H1
        SupportSeek = &H2
        Asynchronous = &H4
    End Enum


    Public NotInheritable Class MSPID
        ' Prevent people from trying to instantiate this class
        Private Sub New()
        End Sub

        Public Shared ReadOnly PrimaryVideo As New Guid(&HA35FF56AUI, &H9FDA, &H11D0, &H8F, &HDF, &H0, _
         &HC0, &H4F, &HD9, &H18, &H9D)

        Public Shared ReadOnly PrimaryAudio As New Guid(&HA35FF56BUI, &H9FDA, &H11D0, &H8F, &HDF, &H0, _
         &HC0, &H4F, &HD9, &H18, &H9D)
    End Class


    <ComImport(), System.Security.SuppressUnmanagedCodeSecurity(), InterfaceType(ComInterfaceType.InterfaceIsIUnknown), Guid("B502D1BD-9A57-11D0-8FDE-00C04FD9189D")> _
    Public Interface IMediaStream
        <PreserveSig()> _
        Function GetMultiMediaStream(<MarshalAs(UnmanagedType.[Interface])> ByRef ppMultiMediaStream As IMultiMediaStream) As Integer

        <PreserveSig()> _
        Function GetInformation(ByRef pPurposeId As Guid, ByRef pType As StreamType) As Integer

        <PreserveSig()> _
        Function SetSameFormat(<[In](), MarshalAs(UnmanagedType.[Interface])> ByVal pStreamThatHasDesiredFormat As IMediaStream, <[In]()> ByVal dwFlags As Integer) As Integer

        <PreserveSig()> _
        Function AllocateSample(<[In]()> ByVal dwFlags As Integer, <MarshalAs(UnmanagedType.[Interface])> ByRef ppSample As IStreamSample) As Integer

        <PreserveSig()> _
        Function CreateSharedSample(<[In](), MarshalAs(UnmanagedType.[Interface])> ByVal pExistingSample As IStreamSample, <[In]()> ByVal dwFlags As Integer, <MarshalAs(UnmanagedType.[Interface])> ByRef ppNewSample As IStreamSample) As Integer

        <PreserveSig()> _
        Function SendEndOfStream(ByVal dwFlags As Integer) As Integer
    End Interface

    <ComImport(), System.Security.SuppressUnmanagedCodeSecurity(), Guid("B502D1BC-9A57-11D0-8FDE-00C04FD9189D"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)> _
    Public Interface IMultiMediaStream
        <PreserveSig()> _
        Function GetInformation(ByRef pdwFlags As MMSSF, ByRef pStreamType As StreamType) As Integer

        <PreserveSig()> _
        Function GetMediaStream(<[In](), MarshalAs(UnmanagedType.LPStruct)> ByVal idPurpose As Guid, <MarshalAs(UnmanagedType.[Interface])> ByRef ppMediaStream As IMediaStream) As Integer

        <PreserveSig()> _
        Function EnumMediaStreams(<[In]()> ByVal Index As Integer, <MarshalAs(UnmanagedType.[Interface])> ByRef ppMediaStream As IMediaStream) As Integer

        <PreserveSig()> _
        Function GetState(ByRef pCurrentState As StreamState) As Integer

        <PreserveSig()> _
        Function SetState(<[In]()> ByVal NewState As StreamState) As Integer

        <PreserveSig()> _
        Function GetTime(ByRef pCurrentTime As Long) As Integer

        <PreserveSig()> _
        Function GetDuration(ByRef pDuration As Long) As Integer

        <PreserveSig()> _
        Function Seek(<[In]()> ByVal SeekTime As Long) As Integer

        <PreserveSig()> _
        Function GetEndOfStreamEventHandle(ByRef phEOS As IntPtr) As Integer
    End Interface

    <ComImport(), System.Security.SuppressUnmanagedCodeSecurity(), Guid("B502D1BE-9A57-11D0-8FDE-00C04FD9189D"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)> _
    Public Interface IStreamSample
        <PreserveSig()> _
        Function GetMediaStream(<MarshalAs(UnmanagedType.[Interface])> ByRef ppMediaStream As IMediaStream) As Integer

        <PreserveSig()> _
        Function GetSampleTimes(ByRef pStartTime As Long, ByRef pEndTime As Long, ByRef pCurrentTime As Long) As Integer

        <PreserveSig()> _
        Function SetSampleTimes(<[In]()> ByVal pStartTime As DsLong, <[In]()> ByVal pEndTime As DsLong) As Integer

        <PreserveSig()> _
        Function Update(<[In]()> ByVal dwFlags As SSUpdate, <[In]()> ByVal hEvent As IntPtr, <[In]()> ByVal pfnAPC As IntPtr, <[In]()> ByVal dwAPCData As IntPtr) As Integer

        <PreserveSig()> _
        Function CompletionStatus(<[In]()> ByVal dwFlags As CompletionStatusFlags, <[In]()> ByVal dwMilliseconds As Integer) As Integer
    End Interface


End Namespace
