
Imports System.Text
Imports System.Runtime.InteropServices
Imports System.Security

Namespace DMO

#If ALLOW_UNTESTED_INTERFACES Then

	<Flags> _
	Public Enum DMOQualityStatus
		None = &H0
		Enabled = &H1
	End Enum

#End If

    <Flags()> _
    Public Enum DMOOutputDataBufferFlags
        None = 0
        SyncPoint = &H1
        Time = &H2
        TimeLength = &H4
        InComplete = &H1000000
    End Enum

    <Flags()> _
    Public Enum DMOEnumerator
        None = 0
        IncludeKeyed = &H1
    End Enum

    <Flags()> _
    Public Enum DMORegisterFlags
        None = 0
        IsKeyed = &H1
    End Enum

    <Flags()> _
    Public Enum DMOProcessOutput
        None = &H0
        DiscardWhenNoBuffer = &H1
    End Enum

    <Flags()> _
    Public Enum DMOInputDataBuffer
        None = 0
        SyncPoint = &H1
        Time = &H2
        TimeLength = &H4
    End Enum

    <Flags()> _
    Public Enum DMOInplaceProcess
        Normal = 0
        Zero = &H1
    End Enum

    <Flags()> _
    Public Enum DMOInputStreamInfo
        None = &H0
        WholeSamples = &H1
        SingleSamplePerBuffer = &H2
        FixedSampleSize = &H4
        HoldsBuffers = &H8
    End Enum

    <Flags()> _
    Public Enum DMOOutputStreamInfo
        None = &H0
        WholeSamples = &H1
        SingleSamplePerBuffer = &H2
        FixedSampleSize = &H4
        Discardable = &H8
        [Optional] = &H10
    End Enum

    <Flags()> _
    Public Enum DMOSetType
        None = &H0
        TestOnly = &H1
        Clear = &H2
    End Enum

    <Flags()> _
    Public Enum DMOInputStatusFlags
        None = &H0
        AcceptData = &H1
    End Enum

    <Flags()> _
    Public Enum DMOVideoOutputStream
        None = &H0
        NeedsPreviousSample = &H1
    End Enum

    <StructLayout(LayoutKind.Sequential)> _
    Public Structure DMOPartialMediatype
        Public type As Guid
        Public subtype As Guid
    End Structure

    <StructLayout(LayoutKind.Sequential, Pack:=8)> _
    Public Structure DMOOutputDataBuffer
        <MarshalAs(UnmanagedType.[Interface])> _
        Public pBuffer As IMediaBuffer
        Public dwStatus As DMOOutputDataBufferFlags
        Public rtTimestamp As Long
        Public rtTimelength As Long
    End Structure


    Public NotInheritable Class DMOCategory
        ' Prevent people from trying to instantiate this class
        Private Sub New()
        End Sub

        Public Shared ReadOnly AudioDecoder As New Guid(&H57F2DB8B, &HE6BB, &H4513, &H9D, &H43, &HDC, _
         &HD2, &HA6, &H59, &H31, &H25)

        Public Shared ReadOnly AudioEncoder As New Guid(&H33D9A761, &H90C8, &H11D0, &HBD, &H43, &H0, _
         &HA0, &HC9, &H11, &HCE, &H86)

        Public Shared ReadOnly VideoDecoder As New Guid(&H4A69B442, &H28BE, &H4991, &H96, &H9C, &HB5, _
         &H0, &HAD, &HF5, &HD8, &HA8)

        Public Shared ReadOnly VideoEncoder As New Guid(&H33D9A760, &H90C8, &H11D0, &HBD, &H43, &H0, _
         &HA0, &HC9, &H11, &HCE, &H86)

        Public Shared ReadOnly AudioEffect As New Guid(&HF3602B3FUI, &H592, &H48DF, &HA4, &HCD, &H67, _
         &H47, &H21, &HE7, &HEB, &HEB)

        Public Shared ReadOnly VideoEffect As New Guid(&HD990EE14UI, &H776C, &H4723, &HBE, &H46, &H3D, _
         &HA2, &HF5, &H6F, &H10, &HB9)

        Public Shared ReadOnly AudioCaptureEffect As New Guid(&HF665AABAUI, &H3E09, &H4920, &HAA, &H5F, &H21, _
         &H98, &H11, &H14, &H8F, &H9)
    End Class


    Public NotInheritable Class DMOUtils
        <DllImport("msdmo.dll", ExactSpelling:=True), SuppressUnmanagedCodeSecurity()> _
        Public Shared Function DMOEnum(<MarshalAs(UnmanagedType.LPStruct)> ByVal DMOCategory As Guid, ByVal dwFlags As DMOEnumerator, ByVal cInTypes As Integer, <[In]()> ByVal pInTypes As DMOPartialMediatype(), ByVal cOutTypes As Integer, <[In]()> ByVal pOutTypes As DMOPartialMediatype(), _
         ByRef ppEnum As IEnumDMO) As Integer
        End Function

        <DllImport("msdmo.dll", ExactSpelling:=True), SuppressUnmanagedCodeSecurity()> _
        Public Shared Function MoInitMediaType(<Out()> ByVal pmt As AMMediaType, ByVal i As Integer) As Integer
        End Function

        <DllImport("msdmo.dll", ExactSpelling:=True), SuppressUnmanagedCodeSecurity()> _
        Public Shared Function MoCopyMediaType(<Out(), MarshalAs(UnmanagedType.LPStruct)> ByVal dst As AMMediaType, <[In](), MarshalAs(UnmanagedType.LPStruct)> ByVal src As AMMediaType) As Integer
        End Function

        <DllImport("MSDmo.dll", CharSet:=CharSet.Unicode, ExactSpelling:=True), SuppressUnmanagedCodeSecurity()> _
        Public Shared Function DMORegister(<MarshalAs(UnmanagedType.LPWStr)> ByVal szName As String, <[In](), MarshalAs(UnmanagedType.LPStruct)> ByVal clsidDMO As Guid, <[In](), MarshalAs(UnmanagedType.LPStruct)> ByVal DMOCategory As Guid, ByVal dwFlags As DMORegisterFlags, ByVal cInTypes As Integer, <[In]()> ByVal pInTypes As DMOPartialMediatype(), _
         ByVal cOutTypes As Integer, <[In]()> ByVal pOutTypes As DMOPartialMediatype()) As Integer
        End Function

        <DllImport("MSDmo.dll", ExactSpelling:=True), SuppressUnmanagedCodeSecurity()> _
        Public Shared Function DMOUnregister(<[In](), MarshalAs(UnmanagedType.LPStruct)> ByVal clsidDMO As Guid, <[In](), MarshalAs(UnmanagedType.LPStruct)> ByVal guidCategory As Guid) As Integer
        End Function

        <DllImport("MSDmo.dll", CharSet:=CharSet.Unicode, ExactSpelling:=True), SuppressUnmanagedCodeSecurity()> _
        Public Shared Function DMOGetName(<[In](), MarshalAs(UnmanagedType.LPStruct)> ByVal clsidDMO As Guid, <Out(), MarshalAs(UnmanagedType.LPWStr, SizeConst:=80)> ByVal szName As StringBuilder) As Integer
        End Function

        <DllImport("MSDmo.dll", ExactSpelling:=True), SuppressUnmanagedCodeSecurity()> _
        Public Shared Function DMOGetTypes(<[In](), MarshalAs(UnmanagedType.LPStruct)> ByVal clsidDMO As Guid, ByVal ulInputTypesRequested As Integer, ByRef pulInputTypesSupplied As Integer, <Out()> ByVal pInTypes As DMOPartialMediatype(), ByVal ulOutputTypesRequested As Integer, ByRef pulOutputTypesSupplied As Integer, _
         <Out()> ByVal pOutTypes As DMOPartialMediatype()) As Integer
        End Function

        Private Sub New()
        End Sub
    End Class



    Public NotInheritable Class DMOResults
        ' Prevent people from trying to instantiate this class
        Private Sub New()
        End Sub

        Public Const E_InvalidStreamIndex As Integer = &H80040201
        Public Const E_InvalidType As Integer = &H80040202
        Public Const E_TypeNotSet As Integer = &H80040203
        Public Const E_NotAccepting As Integer = &H80040204
        Public Const E_TypeNotAccepted As Integer = &H80040205
        Public Const E_NoMoreItems As Integer = &H80040206
    End Class

    Public NotInheritable Class DMOError
        ' Prevent people from trying to instantiate this class
        Private Sub New()
        End Sub

        Public Shared Function GetErrorText(ByVal hr As Integer) As String
            Dim sRet As String = Nothing

            Select Case hr
                Case DMOResults.E_InvalidStreamIndex
                    sRet = "Invalid stream index."
                    Exit Select
                Case DMOResults.E_InvalidType
                    sRet = "Invalid media type."
                    Exit Select
                Case DMOResults.E_TypeNotSet
                    sRet = "Media type was not set. One or more streams require a media type before this operation can be performed."
                    Exit Select
                Case DMOResults.E_NotAccepting
                    sRet = "Data cannot be accepted on this stream. You might need to process more output data; see IMediaObject::ProcessInput."
                    Exit Select
                Case DMOResults.E_TypeNotAccepted
                    sRet = "Media type was not accepted."
                    Exit Select
                Case DMOResults.E_NoMoreItems
                    sRet = "Media-type index is out of range."
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


#If ALLOW_UNTESTED_INTERFACES Then

	<ComImport, System.Security.SuppressUnmanagedCodeSecurity, Guid("65ABEA96-CF36-453F-AF8A-705E98F16260"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)> _
	Public Interface IDMOQualityControl
		<PreserveSig> _
		Function SetNow(<[In]> rtNow As Long) As Integer

		<PreserveSig> _
		Function SetStatus(<[In]> dwFlags As DMOQualityStatus) As Integer

		<PreserveSig> _
		Function GetStatus(ByRef pdwFlags As DMOQualityStatus) As Integer
	End Interface

#End If

    <ComImport(), System.Security.SuppressUnmanagedCodeSecurity(), InterfaceType(ComInterfaceType.InterfaceIsIUnknown), Guid("2C3CD98A-2BFA-4A53-9C27-5249BA64BA0F")> _
    Public Interface IEnumDMO
        <PreserveSig()> _
        Function [Next](ByVal cItemsToFetch As Integer, <Out(), MarshalAs(UnmanagedType.LPArray, SizeParamIndex:=0)> ByVal pCLSID As Guid(), <Out(), MarshalAs(UnmanagedType.LPArray, SizeParamIndex:=0, ArraySubType:=UnmanagedType.LPWStr)> ByVal Names As String(), <[In]()> ByVal pcItemsFetched As IntPtr) As Integer

        <PreserveSig()> _
        Function Skip(ByVal cItemsToSkip As Integer) As Integer

        <PreserveSig()> _
        Function Reset() As Integer

        <PreserveSig()> _
        Function Clone(<MarshalAs(UnmanagedType.[Interface])> ByRef ppEnum As IEnumDMO) As Integer
    End Interface

    <ComImport(), System.Security.SuppressUnmanagedCodeSecurity(), InterfaceType(ComInterfaceType.InterfaceIsIUnknown), Guid("651B9AD0-0FC7-4AA9-9538-D89931010741")> _
    Public Interface IMediaObjectInPlace
        <PreserveSig()> _
        Function Process(<[In]()> ByVal ulSize As Integer, <[In]()> ByVal pData As IntPtr, <[In]()> ByVal refTimeStart As Long, <[In]()> ByVal dwFlags As DMOInplaceProcess) As Integer

        <PreserveSig()> _
        Function Clone(<MarshalAs(UnmanagedType.[Interface])> ByRef ppMediaObject As IMediaObjectInPlace) As Integer

        <PreserveSig()> _
        Function GetLatency(ByRef pLatencyTime As Long) As Integer
    End Interface

    <ComImport(), System.Security.SuppressUnmanagedCodeSecurity(), Guid("59EFF8B9-938C-4A26-82F2-95CB84CDC837"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)> _
    Public Interface IMediaBuffer
        <PreserveSig()> _
        Function SetLength(ByVal cbLength As Integer) As Integer

        <PreserveSig()> _
        Function GetMaxLength(ByRef pcbMaxLength As Integer) As Integer

        <PreserveSig()> _
        Function GetBufferAndLength(ByRef ppBuffer As IntPtr, ByRef pcbLength As Integer) As Integer
    End Interface

    <ComImport(), System.Security.SuppressUnmanagedCodeSecurity(), Guid("D8AD0F58-5494-4102-97C5-EC798E59BCF4"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)> _
    Public Interface IMediaObject
        <PreserveSig()> _
        Function GetStreamCount(ByRef pcInputStreams As Integer, ByRef pcOutputStreams As Integer) As Integer

        <PreserveSig()> _
        Function GetInputStreamInfo(ByVal dwInputStreamIndex As Integer, ByRef pdwFlags As DMOInputStreamInfo) As Integer

        <PreserveSig()> _
        Function GetOutputStreamInfo(ByVal dwOutputStreamIndex As Integer, ByRef pdwFlags As DMOOutputStreamInfo) As Integer

        <PreserveSig()> _
        Function GetInputType(ByVal dwInputStreamIndex As Integer, ByVal dwTypeIndex As Integer, <Out()> ByVal pmt As AMMediaType) As Integer

        <PreserveSig()> _
        Function GetOutputType(ByVal dwOutputStreamIndex As Integer, ByVal dwTypeIndex As Integer, <Out()> ByVal pmt As AMMediaType) As Integer

        <PreserveSig()> _
        Function SetInputType(ByVal dwInputStreamIndex As Integer, <[In](), MarshalAs(UnmanagedType.LPStruct)> ByVal pmt As AMMediaType, ByVal dwFlags As DMOSetType) As Integer

        <PreserveSig()> _
        Function SetOutputType(ByVal dwOutputStreamIndex As Integer, <[In](), MarshalAs(UnmanagedType.LPStruct)> ByVal pmt As AMMediaType, ByVal dwFlags As DMOSetType) As Integer

        <PreserveSig()> _
        Function GetInputCurrentType(ByVal dwInputStreamIndex As Integer, <Out()> ByVal pmt As AMMediaType) As Integer

        <PreserveSig()> _
        Function GetOutputCurrentType(ByVal dwOutputStreamIndex As Integer, <Out()> ByVal pmt As AMMediaType) As Integer

        <PreserveSig()> _
        Function GetInputSizeInfo(ByVal dwInputStreamIndex As Integer, ByRef pcbSize As Integer, ByRef pcbMaxLookahead As Integer, ByRef pcbAlignment As Integer) As Integer

        <PreserveSig()> _
        Function GetOutputSizeInfo(ByVal dwOutputStreamIndex As Integer, ByRef pcbSize As Integer, ByRef pcbAlignment As Integer) As Integer

        <PreserveSig()> _
        Function GetInputMaxLatency(ByVal dwInputStreamIndex As Integer, ByRef prtMaxLatency As Long) As Integer

        <PreserveSig()> _
        Function SetInputMaxLatency(ByVal dwInputStreamIndex As Integer, ByVal rtMaxLatency As Long) As Integer

        <PreserveSig()> _
        Function Flush() As Integer

        <PreserveSig()> _
        Function Discontinuity(ByVal dwInputStreamIndex As Integer) As Integer

        <PreserveSig()> _
        Function AllocateStreamingResources() As Integer

        <PreserveSig()> _
        Function FreeStreamingResources() As Integer

        <PreserveSig()> _
        Function GetInputStatus(ByVal dwInputStreamIndex As Integer, ByRef dwFlags As DMOInputStatusFlags) As Integer

        <PreserveSig()> _
        Function ProcessInput(ByVal dwInputStreamIndex As Integer, ByVal pBuffer As IMediaBuffer, ByVal dwFlags As DMOInputDataBuffer, ByVal rtTimestamp As Long, ByVal rtTimelength As Long) As Integer

        <PreserveSig()> _
        Function ProcessOutput(ByVal dwFlags As DMOProcessOutput, ByVal cOutputBufferCount As Integer, <[In](), Out(), MarshalAs(UnmanagedType.LPArray, SizeParamIndex:=1)> ByVal pOutputBuffers As DMOOutputDataBuffer(), ByRef pdwStatus As Integer) As Integer

        <PreserveSig()> _
        Function Lock(<MarshalAs(UnmanagedType.Bool)> ByVal bLock As Boolean) As Integer
    End Interface

    <ComImport(), System.Security.SuppressUnmanagedCodeSecurity(), Guid("BE8F4F4E-5B16-4D29-B350-7F6B5D9298AC"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)> _
    Public Interface IDMOVideoOutputOptimizations
        <PreserveSig()> _
        Function QueryOperationModePreferences(ByVal ulOutputStreamIndex As Integer, ByRef pdwRequestedCapabilities As DMOVideoOutputStream) As Integer

        <PreserveSig()> _
        Function SetOperationMode(ByVal ulOutputStreamIndex As Integer, ByVal dwEnabledFeatures As DMOVideoOutputStream) As Integer

        <PreserveSig()> _
        Function GetCurrentOperationMode(ByVal ulOutputStreamIndex As Integer, ByRef pdwEnabledFeatures As DMOVideoOutputStream) As Integer

        <PreserveSig()> _
        Function GetCurrentSampleRequirements(ByVal ulOutputStreamIndex As Integer, ByRef pdwRequestedFeatures As DMOVideoOutputStream) As Integer
    End Interface

End Namespace
