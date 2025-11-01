
Imports System.IO

Imports System.Text
Imports System.Runtime.InteropServices
Imports System.Diagnostics

Public Module DirecShow_Utils


    Public Function Show_SourceDialog_VFW(ByVal CaptureGraphBuilder As ICaptureGraphBuilder2, _
                                          ByVal captureFilter As IBaseFilter, _
                                          ByVal hwnd As IntPtr) As Boolean

        Dim comObj As Object = Nothing
        ''
        'Dim iid As Guid = GetType(IAMVfwCaptureDialogs).GUID
        'CaptureGraphBuilder.FindInterface(Nothing, Nothing, captureFilter, iid, comObj)
        'Dim VfwDialogs As IAMVfwCaptureDialogs = CType(comObj, IAMVfwCaptureDialogs)
        ''
        'If VfwDialogs IsNot Nothing Then
        '    VfwDialogs.ShowDialog(VfwCaptureDialogs.Source, hwnd)
        '    Show_SourceDialog_VFW = True
        'Else
        '    Show_SourceDialog_VFW = False
        'End If
        '
        If comObj IsNot Nothing Then Marshal.ReleaseComObject(comObj) : comObj = Nothing
    End Function

    Public Function Show_FormatDialog_VFW(ByVal CaptureGraphBuilder As ICaptureGraphBuilder2, _
                                          ByVal captureFilter As IBaseFilter, _
                                          ByVal hwnd As IntPtr) As Boolean

        Dim comObj As Object = Nothing
        '
        'CaptureGraphBuilder.FindInterface(Nothing, Nothing, captureFilter, GetType(IAMVfwCaptureDialogs).GUID, comObj)
        'Dim VfwDialogs As IAMVfwCaptureDialogs = CType(comObj, IAMVfwCaptureDialogs)
        ''
        'If VfwDialogs IsNot Nothing Then
        '    VfwDialogs.ShowDialog(VfwCaptureDialogs.Format, hwnd)
        '    Show_FormatDialog_VFW = True
        'Else
        '    Show_FormatDialog_VFW = False
        'End If
        '
        If comObj IsNot Nothing Then Marshal.ReleaseComObject(comObj) : comObj = Nothing
    End Function



    Public Function Show_SourceDialog_WDM(ByVal CaptureGraphBuilder As ICaptureGraphBuilder2, _
                                          ByVal captureFilter As IBaseFilter, _
                                          ByVal hwnd As IntPtr) As Boolean

        Dim comObj As Object = Nothing
        Dim cauuid As New DsCAUUID()
        '
        '                                                                         IAMVideoControl or IAMVideoProcAmp  
        CaptureGraphBuilder.FindInterface(PIN_CATEGORY_CAPTURE, Nothing, captureFilter, GetType(IAMVideoControl).GUID, comObj)
        Dim spec As ISpecifyPropertyPages = TryCast(comObj, ISpecifyPropertyPages)
        '
        If spec IsNot Nothing Then
            spec.GetPages(cauuid)
            OleCreatePropertyFrame(hwnd, 0, 0, "WDM video input control", 1, comObj, cauuid.cElems, cauuid.pElems, 0, 0, IntPtr.Zero)
            Show_SourceDialog_WDM = True
        Else
            Show_SourceDialog_WDM = False
        End If
        '
        If cauuid.pElems <> IntPtr.Zero Then Marshal.FreeCoTaskMem(cauuid.pElems) : spec = Nothing
        If comObj IsNot Nothing Then Marshal.ReleaseComObject(comObj) : comObj = Nothing
    End Function

    Public Function Show_FormatDialog_WDM(ByVal CaptureGraphBuilder As ICaptureGraphBuilder2, _
                                          ByVal captureFilter As IBaseFilter, _
                                          ByVal hwnd As IntPtr) As Boolean

        Dim comObj As Object = Nothing
        Dim cauuid As New DsCAUUID()
        '
        CaptureGraphBuilder.FindInterface(PIN_CATEGORY_CAPTURE, Nothing, captureFilter, GetType(IAMStreamConfig).GUID, comObj)
        '
        If comobj IsNot Nothing Then
            TryCast(comObj, ISpecifyPropertyPages).GetPages(cauuid)
            OleCreatePropertyFrame(hwnd, 30, 30, "", 1, comObj, cauuid.cElems, cauuid.pElems, 0, 0, IntPtr.Zero)
            Show_FormatDialog_WDM = True
        Else
            Show_FormatDialog_WDM = False
        End If
        '
        If cauuid.pElems <> IntPtr.Zero Then Marshal.FreeCoTaskMem(cauuid.pElems)
        If comObj IsNot Nothing Then Marshal.ReleaseComObject(comObj) : comObj = Nothing
    End Function



    'Public Function Show_TunerPinDialog_WDM(ByVal CaptureGraphBuilder As ICaptureGraphBuilder2, _
    '                                        ByVal captureFilter As IBaseFilter, _
    '                                        ByVal hwnd As IntPtr) As Boolean

    '    Dim hr As Integer
    '    Dim comObj As Object = Nothing
    '    Dim spec As ISpecifyPropertyPages = Nothing
    '    Dim cauuid As New DsCAUUID()

    '    Try
    '        Dim cat As Guid = PinCategory.Capture
    '        Dim type As Guid = MediaType.Interleaved
    '        Dim iid As Guid = GetType(IAMTVTuner).GUID
    '        hr = CaptureGraphBuilder.FindInterface(cat, type, captureFilter, iid, comObj)
    '        If hr <> 0 Then
    '            type = MediaType.Video
    '            hr = CaptureGraphBuilder.FindInterface(cat, type, captureFilter, iid, comObj)
    '            If hr <> 0 Then
    '                Return False
    '            End If
    '        End If
    '        spec = TryCast(comObj, ISpecifyPropertyPages)
    '        If spec Is Nothing Then
    '            Return False
    '        End If

    '        hr = spec.GetPages(cauuid)
    '        hr = OleCreatePropertyFrame(hwnd, 30, 30, Nothing, 1, comObj, _
    '                                    cauuid.cElems, cauuid.pElems, 0, 0, IntPtr.Zero)
    '        Return True

    '    Catch ee As Exception
    '        Trace.WriteLine("!Ds.NET: ShowCapPinDialog " & ee.Message)
    '        Return False

    '    Finally
    '        If cauuid.pElems <> IntPtr.Zero Then Marshal.FreeCoTaskMem(cauuid.pElems)
    '        spec = Nothing
    '        If comObj IsNot Nothing Then Marshal.ReleaseComObject(comObj)
    '        comObj = Nothing

    '    End Try
    'End Function




    <DllImport("olepro32.dll", CharSet:=CharSet.Unicode, ExactSpelling:=True)> _
    Private Function OleCreatePropertyFrame(ByVal hwndOwner As IntPtr, _
                                                   ByVal x As Integer, _
                                                   ByVal y As Integer, _
                                                   ByVal lpszCaption As String, _
                                                   ByVal cObjects As Integer, _
                                                   <[In](), MarshalAs(UnmanagedType.[Interface])> ByRef ppUnk As Object, _
                                                   ByVal cPages As Integer, _
                                                   ByVal pPageClsID As IntPtr, _
                                                   ByVal lcid As Integer, _
                                                   ByVal dwReserved As Integer, _
                                                   ByVal pvReserved As IntPtr) As Integer
    End Function


    ' Not working formats
    ' ----------------------------------------
    'YUY2	0x32595559	16	YUV 4:2:2 as for UYVY but with different ordering in the u_int32 macropixel
    'NV12	0x3231564E	12	8-bit Y plane followed by an interleaved U/V plane with 2x2 subsampling


    Friend Function MediaSubTypeToString(ByVal guid As Guid) As String
        Select Case LCase(guid.ToString)
            Case "e436eb78-524f-11ce-9f53-0020af0ba770" : Return "RGB1"
            Case "e436eb79-524f-11ce-9f53-0020af0ba770" : Return "RGB4"
            Case "e436eb7a-524f-11ce-9f53-0020af0ba770" : Return "RGB8"
            Case "e436eb7d-524f-11ce-9f53-0020af0ba770" : Return "RGB24"
            Case "e436eb7e-524f-11ce-9f53-0020af0ba770" : Return "RGB32"
            Case "e436eb7c-524f-11ce-9f53-0020af0ba770" : Return "RGB555"
            Case "e436eb7b-524f-11ce-9f53-0020af0ba770" : Return "RGB565"
            Case "30323449-0000-0010-8000-00aa00389b71" : Return "I420"
            Case "56555949-0000-0010-8000-00aa00389b71" : Return "IYUV"
            Case "59565955-0000-0010-8000-00aa00389b71" : Return "UYVY"
            Case "32595559-0000-0010-8000-00aa00389b71" : Return "YUY2"
            Case "31313259-0000-0010-8000-00aa00389b71" : Return "YV12"
            Case "3231564e-0000-0010-8000-00aa00389b71" : Return "NV12"
            Case "39555659-0000-0010-8000-00aa00389b71" : Return "YVU9"
            Case "55595659-0000-0010-8000-00aa00389b71" : Return "YVYU"
            Case "47504a4d-0000-0010-8000-00aa00389b71" : Return "MJPG"
        End Select
        Return ""
    End Function

    Friend Function MediaSubTypeFromString(ByVal StringMediaSubType As String) As Guid
        Dim g As String
        Select Case StringMediaSubType
            Case "RGB1" : g = "{e436eb78-524f-11ce-9f53-0020af0ba770}"
            Case "RGB4" : g = "{e436eb79-524f-11ce-9f53-0020af0ba770}"
            Case "RGB8" : g = "{e436eb7a-524f-11ce-9f53-0020af0ba770}"
            Case "RGB24" : g = "{e436eb7d-524f-11ce-9f53-0020af0ba770}"
            Case "RGB32" : g = "{e436eb7e-524f-11ce-9f53-0020af0ba770}"
            Case "RGB555" : g = "{e436eb7c-524f-11ce-9f53-0020af0ba770}"
            Case "RGB565" : g = "{e436eb7b-524f-11ce-9f53-0020af0ba770}"
            Case "I420" : g = "{30323449-0000-0010-8000-00aa00389b71}"
            Case "IYUV" : g = "{56555949-0000-0010-8000-00aa00389b71}"
            Case "UYVY" : g = "{59565955-0000-0010-8000-00aa00389b71}"
            Case "YUY2" : g = "{32595559-0000-0010-8000-00aa00389b71}"
            Case "YV12" : g = "{31313259-0000-0010-8000-00aa00389b71}"
            Case "NV12" : g = "{3231564e-0000-0010-8000-00aa00389b71}"
            Case "YVU9" : g = "{39555659-0000-0010-8000-00aa00389b71}"
            Case "YVYU" : g = "{55595659-0000-0010-8000-00aa00389b71}"
            Case "MJPG" : g = "{47504a4d-0000-0010-8000-00aa00389b71}"
            Case Else : g = "{00000000-0000-0000-0000-000000000000}"
        End Select
        Return New Guid(g)
    End Function

End Module
