
Imports System.Runtime.InteropServices

Namespace DMO

    Public NotInheritable Class MediaParamTimeFormat
        ' Prevent people from trying to instantiate this class
        Private Sub New()
        End Sub

        Public Shared ReadOnly Reference As New Guid(&H93AD712BUI, &HDAA0, &H4FFE, &HBC, &H81, &HB0, _
         &HCE, &H50, &HF, &HCD, &HD9)

        Public Shared ReadOnly Music As New Guid(&H574C49D, &H5B04, &H4B15, &HA5, &H42, &HAE, _
         &H28, &H20, &H30, &H11, &H7B)

        Public Shared ReadOnly Samples As New Guid(&HA8593D05UI, &HC43, &H4984, &H9A, &H63, &H97, _
         &HAF, &H9E, &H2, &HC4, &HC0)
    End Class

    <StructLayout(LayoutKind.Explicit)> _
    Public Structure MPData
        <FieldOffset(0)> _
        Public vBool As Boolean
        <FieldOffset(0)> _
        Public vFloat As Single
        <FieldOffset(0)> _
        Public vInt As Integer
    End Structure

    <StructLayout(LayoutKind.Sequential, Pack:=8)> _
    Public Structure MPEnvelopeSegment
        Public rtStart As Long
        Public rtEnd As Long
        Public valStart As MPData
        Public valEnd As MPData
        Public iCurve As MPCaps
        Public flags As MPFlags
    End Structure

    <Flags()> _
    Public Enum MPFlags
        Standard = &H0
        BeginCurrentVal = &H1
        BeginNeutralVal = &H2
    End Enum

    Public Enum MPType
        ' Fields
        BOOL = 2
        [ENUM] = 3
        FLOAT = 1
        INT = 0
        MAX = 4
    End Enum

    <Flags()> _
    Public Enum MPCaps
        None = 0
        Jump = &H1
        Linear = &H2
        Square = &H4
        InvSquare = &H8
        Sine = &H10
    End Enum

    <StructLayout(LayoutKind.Sequential, Pack:=4, CharSet:=CharSet.Unicode)> _
    Public Structure ParamInfo
        Public mpType As MPType
        Public mopCaps As MPCaps
        Public mpdMinValue As MPData
        Public mpdMaxValue As MPData
        Public mpdNeutralValue As MPData
        <MarshalAs(UnmanagedType.ByValTStr, SizeConst:=&H20)> _
        Public szUnitText As String
        <MarshalAs(UnmanagedType.ByValTStr, SizeConst:=&H20)> _
        Public szLabel As String
    End Structure


    <ComImport(), System.Security.SuppressUnmanagedCodeSecurity(), InterfaceType(ComInterfaceType.InterfaceIsIUnknown), Guid("6D6CBB60-A223-44AA-842F-A2F06750BE6D")> _
    Public Interface IMediaParamInfo
        <PreserveSig()> _
        Function GetParamCount(ByRef pdwParams As Integer) As Integer

        <PreserveSig()> _
        Function GetParamInfo(<[In]()> ByVal dwParamIndex As Integer, ByRef pInfo As ParamInfo) As Integer

        <PreserveSig()> _
        Function GetParamText(<[In]()> ByVal dwParamIndex As Integer, ByRef ip As IntPtr) As Integer

        <PreserveSig()> _
        Function GetNumTimeFormats(ByRef pdwNumTimeFormats As Integer) As Integer

        <PreserveSig()> _
        Function GetSupportedTimeFormat(<[In]()> ByVal dwFormatIndex As Integer, ByRef pguidTimeFormat As Guid) As Integer

        <PreserveSig()> _
        Function GetCurrentTimeFormat(ByRef pguidTimeFormat As Guid, ByRef pTimeData As Integer) As Integer
    End Interface

    <ComImport(), System.Security.SuppressUnmanagedCodeSecurity(), InterfaceType(ComInterfaceType.InterfaceIsIUnknown), Guid("6D6CBB61-A223-44AA-842F-A2F06750BE6E")> _
    Public Interface IMediaParams
        <PreserveSig()> _
        Function GetParam(<[In]()> ByVal dwParamIndex As Integer, ByRef pValue As MPData) As Integer

        <PreserveSig()> _
        Function SetParam(<[In]()> ByVal dwParamIndex As Integer, <[In]()> ByVal value As MPData) As Integer

        <PreserveSig()> _
        Function AddEnvelope(<[In]()> ByVal dwParamIndex As Integer, <[In]()> ByVal cSegments As Integer, <[In](), MarshalAs(UnmanagedType.LPArray, SizeParamIndex:=1)> ByVal pEnvelopeSegments As MPEnvelopeSegment()) As Integer

        <PreserveSig()> _
        Function FlushEnvelope(<[In]()> ByVal dwParamIndex As Integer, <[In]()> ByVal refTimeStart As Long, <[In]()> ByVal refTimeEnd As Long) As Integer

        <PreserveSig()> _
        Function SetTimeFormat(<[In]()> ByVal MediaParamTimeFormat As Guid, <[In]()> ByVal mpTimeData As Integer) As Integer
    End Interface

End Namespace
