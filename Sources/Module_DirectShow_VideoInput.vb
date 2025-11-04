
Imports System.Runtime.InteropServices
Imports System.Drawing

Module Module_DirectShow_VideoInput

    Friend Capture_Image As Image
    Friend Capture_FramesPerSecond As Double
    Friend Capture_NewImageIsReady As Boolean = False

    Private GraphBuilder As IFilterGraph2 = Nothing
    Private CaptureGraphBuilder As ICaptureGraphBuilder2 = Nothing
    Private CaptureFilter As IBaseFilter = Nothing
    Friend MediaControl As IMediaControl = Nothing
    Private sampleGrabber As IBaseFilter
    Private sgcb As SamplegrabberCallback = New SamplegrabberCallback

    Private rot As DsROTEntry = Nothing

    Friend Sub Capture_INIT(ByVal CaptureDeviceName As String)
        Try
            ' --------------------------------------------------------------------- GRAPH BUILDER
            GraphBuilder = CType(New FilterGraph, IFilterGraph2)
            MediaControl = CType(GraphBuilder, IMediaControl)
            CaptureGraphBuilder = CType(New CaptureGraphBuilder2, ICaptureGraphBuilder2)
            CaptureGraphBuilder.SetFiltergraph(GraphBuilder)

            ' --------------------------------------------------------------------- GRAPH Visible with GraphEdit
            rot = New DsROTEntry(GraphBuilder)

            ' --------------------------------------------------------------------- CAPTURE DEVICE
            SelectCaptureDevice(CaptureDeviceName)
            '
            If CaptureFilter Is Nothing Or CaptureDeviceName = "" Then
                Capture_STOP()
                Capture_Image = Nothing
                Exit Sub
            End If

        Catch ex As Exception
            MsgBox("Can not open: " & CaptureDeviceName, MsgBoxStyle.Information)
            Capture_STOP()
            Capture_Image = Nothing
        End Try
    End Sub

    Friend Sub Capture_ConnectFiltersAndRun()
        Try
            Release_IbaseFilter(sampleGrabber)

            ' --------------------------------------------------------------------- SAMPLE GRABBER
            sampleGrabber = DirectCast(New SampleGrabber(), IBaseFilter)
            GraphBuilder.AddFilter(sampleGrabber, "Sample Grabber")

            ' --------------------------------------------------------------------- PREVIEW ONLY
            'CaptureGraphBuilder.RenderStream(PinCategory.Preview, MediaType.Video, _
            '                                     CaptureFilter, sampleGrabber, Nothing)

            ' --------------------------------------------------------------------- NORMAL
            ConfigureSampleGrabber(DirectCast(sampleGrabber, ISampleGrabber))

            CaptureGraphBuilder.RenderStream(PinCategory.Capture, MediaType.Video, _
                                                  CaptureFilter, Nothing, sampleGrabber)

            ' --------------------------------------------------------------------- START
            MediaControl.Run()

            ' USING BmiHeader - Needs a accurate test
            'Capture_GetVideoFormatParams()

        Catch ex As Exception
            'MsgBox(ex.ToString)
            Capture_STOP()
        End Try
    End Sub

    Private Sub ConfigureSampleGrabber(ByVal sampleGrabber As ISampleGrabber)
        '
        ' ----------------------------------------------------- media type
        Dim media As AMMediaType = New AMMediaType()
        media.majorType = MediaType.Video
        media.subType = MediaSubType.RGB24
        media.formatType = FormatType.VideoInfo
        sampleGrabber.SetMediaType(media)
        DsUtils.FreeAMMediaType(media)
        'sampleGrabber.SetOneShot(False)
        '
        ' ---------------------------------------------------- callback
        sampleGrabber.SetCallback(sgcb, 1) ' 0 = SampleCB / 1 = BufferCB
    End Sub

    Friend Sub Capture_STOP()
        On Error Resume Next

        ' ----------------------------------------------------------- stop previewing data
        If Not MediaControl Is Nothing Then
            MediaControl.StopWhenReady()
        End If

        ' ----------------------------------------------------------- Remove filter graph from the running object table
        If Not rot Is Nothing Then rot.Dispose() : rot = Nothing

        ' ----------------------------------------------------------- Release DirectShow interfaces
        'On Error Resume Next
        '
        Release_IbaseFilter(CaptureFilter)
        Release_IbaseFilter(sampleGrabber)
        '
        ReleaseComObject(CaptureGraphBuilder) : CaptureGraphBuilder = Nothing
        ReleaseComObject(GraphBuilder) : GraphBuilder = Nothing

        Capture_NewImageIsReady = False
    End Sub


    ' ======================================================================================================
    '   SELECT CAPTURE DEVICE
    ' ======================================================================================================
    Private Sub SelectCaptureDevice(ByVal CaptureDeviceName As String)
        '
        Dim fnames() As String = EnumFiltersByCategory(FilterCategory.VideoInputDevice)

        ' --------------------------------------------------------------- Select a named capture device 
        For Each fltName As String In fnames
            If InStr(fltName, CaptureDeviceName, CompareMethod.Text) > 0 Then
                CaptureFilter = GraphTools.AddFilterByName(GraphBuilder, FilterCategory.VideoInputDevice, fltName)
                Exit For
            End If
        Next
    End Sub
    Friend Function VideoCaptureDevice_Is_VFW() As Boolean
        If InStr(VideoInDevice, "vfw", CompareMethod.Text) > 0 Then
            Return True
        Else
            Return False
        End If
    End Function


    ' ====================================================================================================
    '  SHOW DIALOGS
    ' ====================================================================================================
    Friend Sub ShowCaptureFilterDialog(ByVal WindowHandle As IntPtr)
        If CaptureFilter Is Nothing Then Exit Sub
        '
        If Not Show_SourceDialog_WDM(CaptureGraphBuilder, CaptureFilter, WindowHandle) Then
            MediaControl.StopWhenReady()
            Capture_Image = Nothing
            Show_SourceDialog_VFW(CaptureGraphBuilder, CaptureFilter, WindowHandle)
            MediaControl.Run()
        End If

        'Capture_GetVideoParams()
        'Save_INI()
    End Sub

    Friend Sub ShowCapturePinDialog(ByVal WindowHandle As IntPtr)
        If CaptureFilter Is Nothing Then Exit Sub
        ' --------------------------------------------------------- stop and remove filters
        StopAndRemoveAllFilters()
        ' --------------------------------------------------------- open dialog
        '
        If Not Show_FormatDialog_VFW(CaptureGraphBuilder, CaptureFilter, WindowHandle) Then
            Show_FormatDialog_WDM(CaptureGraphBuilder, CaptureFilter, WindowHandle)
        End If
        ' --------------------------------------------------------- reconnect filters and start
        Capture_ConnectFiltersAndRun()
        '
        Capture_GetVideoFormatParams()
        Save_INI()
    End Sub

    Private Sub StopAndRemoveAllFilters()
        ' --------------------------------------------------------- 
        MediaControl.StopWhenReady()
        Capture_NewImageIsReady = False
        Form_Main.PictureBox1_Clear()

        Dim flt As IBaseFilter = Nothing
        GraphBuilder.FindFilterByName("Smart Tee", flt)
        GraphBuilder.RemoveFilter(flt)
        GraphBuilder.FindFilterByName("AVI Decompressor", flt)
        GraphBuilder.RemoveFilter(flt)
        GraphBuilder.FindFilterByName("Sample Grabber", flt)
        GraphBuilder.RemoveFilter(flt)
    End Sub


    ' ====================================================================================================
    '  GET and SET VideoInputParams
    ' ====================================================================================================
    Friend VideoInputParams As Capture_VideoInputParams
    Friend VideoFormatParams As Capture_VideoFormatParams
    '
    Friend Structure Capture_VideoInputParams
        Dim Exposure As Capture_VideoInputParam
        Dim Gain As Capture_VideoInputParam
        Dim Brightness As Capture_VideoInputParam
        Dim Contrast As Capture_VideoInputParam
        Dim Gamma As Capture_VideoInputParam
        Dim BackLight As Capture_VideoInputParam
        '
        Dim Saturation As Capture_VideoInputParam
        Dim WhiteBalance As Capture_VideoInputParam
        Dim Hue As Capture_VideoInputParam
        Dim ColorEnable As Capture_VideoInputParam
        '
        Dim Zoom As Capture_VideoInputParam
        Dim Pan As Capture_VideoInputParam
        Dim Tilt As Capture_VideoInputParam
        '
        Dim Sharpness As Capture_VideoInputParam
        Dim Focus As Capture_VideoInputParam
    End Structure

    Friend Structure Capture_VideoFormatParams
        Dim VideoFormat As String
        Dim VideoSize As String
        Dim VideoFPS As String
    End Structure
    '
    Friend Structure Capture_VideoInputParam
        Dim _Value As Int32
        Dim _Min As Int32
        Dim _Max As Int32
        Dim _Step As Int32
        Dim _Default As Int32
        Dim _AutoValid As Boolean
        Dim _ManualValid As Boolean
        Dim _AutoChecked As Boolean
    End Structure

    Friend Function Capture_GetIAMCameraControl() As IAMCameraControl
        Dim comObj As Object = Nothing
        CaptureGraphBuilder.FindInterface(Nothing, Nothing, CaptureFilter, GetType(IAMCameraControl).GUID, comObj)
        Return TryCast(comObj, IAMCameraControl)
    End Function
    Friend Function Capture_GetIAMVideoProcAmp() As IAMVideoProcAmp
        Dim comObj As Object = Nothing
        CaptureGraphBuilder.FindInterface(Nothing, Nothing, CaptureFilter, GetType(IAMVideoProcAmp).GUID, comObj)
        Return TryCast(comObj, IAMVideoProcAmp)
    End Function
    Friend Function Capture_GetIAMStreamConfig() As IAMStreamConfig
        Dim comObj As Object = Nothing
        CaptureGraphBuilder.FindInterface(Nothing, Nothing, CaptureFilter, GetType(IAMStreamConfig).GUID, comObj)
        Return TryCast(comObj, IAMStreamConfig)
    End Function

    Friend Function Capture_GetCameraControlParam(ByVal prop As CameraControlProperty) As Capture_VideoInputParam
        If CaptureFilter Is Nothing Then Return Nothing
        Dim CameraControl As IAMCameraControl = Capture_GetIAMCameraControl()
        If CameraControl Is Nothing Then Return Nothing
        Dim p As Capture_VideoInputParam
        Dim CapsFlags As CameraControlFlags
        Dim autoChecked As CameraControlFlags
        CameraControl.GetRange(prop, p._Min, p._Max, p._Step, p._Default, CapsFlags)
        CameraControl.Get(prop, p._Value, autoChecked)
        p._AutoValid = (CapsFlags And VideoProcAmpFlags.Auto) = VideoProcAmpFlags.Auto
        p._ManualValid = (CapsFlags And VideoProcAmpFlags.Manual) = VideoProcAmpFlags.Manual
        p._AutoChecked = autoChecked = CameraControlFlags.Auto
        Return p
    End Function
    '
    Friend Sub Capture_SetCameraControlParam(ByVal prop As CameraControlProperty, ByVal param As Capture_VideoInputParam)
        If CaptureFilter Is Nothing Then Exit Sub
        Dim CameraControl As IAMCameraControl = Capture_GetIAMCameraControl()
        If CameraControl Is Nothing Then Exit Sub
        If param._AutoChecked Then
            CameraControl.Set(prop, param._Value, CameraControlFlags.Auto) '前值CameraControlFlags.Auto
        Else
            CameraControl.Set(prop, param._Value, CameraControlFlags.Manual)
        End If
    End Sub

    Friend Function Capture_GetVideoProcAmpParam(ByVal prop As VideoProcAmpProperty) As Capture_VideoInputParam
        If CaptureFilter Is Nothing Then Return Nothing
        Dim VideoProcAmp As IAMVideoProcAmp = Capture_GetIAMVideoProcAmp()
        If VideoProcAmp Is Nothing Then Return Nothing
        Dim p As Capture_VideoInputParam
        Dim CapsFlags As VideoProcAmpFlags
        Dim autoChecked As VideoProcAmpFlags
        VideoProcAmp.GetRange(prop, p._Min, p._Max, p._Step, p._Default, CapsFlags)
        VideoProcAmp.Get(prop, p._Value, autoChecked)
        p._AutoValid = (CapsFlags And VideoProcAmpFlags.Auto) = VideoProcAmpFlags.Auto
        p._ManualValid = (CapsFlags And VideoProcAmpFlags.Manual) = VideoProcAmpFlags.Manual
        p._AutoChecked = autoChecked = VideoProcAmpFlags.Auto
        Return p
    End Function
    '
    Friend Sub Capture_SetVideoProcAmpParam(ByVal prop As VideoProcAmpProperty, ByVal param As Capture_VideoInputParam)
        If CaptureFilter Is Nothing Then Exit Sub
        Dim VideoProcAmp As IAMVideoProcAmp = Capture_GetIAMVideoProcAmp()
        If VideoProcAmp Is Nothing Then Exit Sub
        If param._AutoChecked Then
            VideoProcAmp.Set(prop, param._Value, VideoProcAmpFlags.Auto) '前值VideoProcAmpFlags.Auto,调整这里没用
        Else
            VideoProcAmp.Set(prop, param._Value, VideoProcAmpFlags.Manual)
        End If
    End Sub

    Friend Sub Capture_GetVideoFormatParams()
        '
        If VideoCaptureDevice_Is_VFW() Then Exit Sub
        If CaptureFilter Is Nothing Then Exit Sub

        ' ----------------------------------------------------- get IAMStreamConfig
        Dim StreamConfig As IAMStreamConfig = Capture_GetIAMStreamConfig()
        If StreamConfig Is Nothing Then Exit Sub

        ' ----------------------------------------------------- read AMmedia
        Dim AMMedia As AMMediaType = New AMMediaType()
        StreamConfig.GetFormat(AMMedia)
        ' ----------------------------------------------------- read VideoInfoHeader
        Dim vih As VideoInfoHeader = New VideoInfoHeader
        Marshal.PtrToStructure(AMMedia.formatPtr, vih)

        ' ----------------------------------------------------- set SampleGrabbeeCallBack Width and Height 
        ' USING BmiHeader - Needs a accurate test
        sgcb.Width = vih.BmiHeader.Width
        sgcb.Height = vih.BmiHeader.Height

        ' ----------------------------------------------------- MEDIA SUBTYPE
        VideoFormatParams.VideoFormat = MediaSubTypeToString(AMMedia.subType)

        ' ----------------------------------------------------- SIZE
        VideoFormatParams.VideoSize = vih.BmiHeader.Width.ToString & " x " & vih.BmiHeader.Height.ToString

        ' ----------------------------------------------------- FPS
        VideoFormatParams.VideoFPS = (CInt(10000000 / vih.AvgTimePerFrame)).ToString

        ' ----------------------------------------------------- free AMmedia
        DsUtils.FreeAMMediaType(AMMedia)
    End Sub
    '
    Friend Sub Capture_SetVideoFormatParams()
        '
        If VideoCaptureDevice_Is_VFW() Then Exit Sub
        If CaptureFilter Is Nothing Then Exit Sub

        ' ----------------------------------------------------- 
        StopAndRemoveAllFilters()

        ' ----------------------------------------------------- get IAMStreamConfig
        Dim StreamConfig As IAMStreamConfig = Capture_GetIAMStreamConfig()
        If StreamConfig Is Nothing Then Exit Sub

        ' ----------------------------------------------------- read AMmedia
        Dim AMmedia As AMMediaType = New AMMediaType()
        StreamConfig.GetFormat(AMmedia)
        ' ----------------------------------------------------- read VideoInfoHeader
        Dim vih As VideoInfoHeader = New VideoInfoHeader
        Marshal.PtrToStructure(AMmedia.formatPtr, vih)

        ' ----------------------------------------------------- MEDIA SUBTYPE
        AMmedia.subType = MediaSubTypeFromString(VideoFormatParams.VideoFormat)

        ' ----------------------------------------------------- SIZE
        Dim sizeWH() As String = Split(VideoFormatParams.VideoSize, " ")
        If sizeWH.Count = 3 Then
            vih.BmiHeader.Width = CInt(Val(sizeWH(0)))
            vih.BmiHeader.Height = CInt(Val(sizeWH(2)))
        End If

        ' ----------------------------------------------------- FPS
        vih.AvgTimePerFrame = CLng(10000000 / Val(VideoFormatParams.VideoFPS))

        ' ----------------------------------------------------- write VideoInfoHeader
        Marshal.StructureToPtr(vih, AMmedia.formatPtr, True)
        ' ----------------------------------------------------- write AMmedia
        StreamConfig.SetFormat(AMmedia)
        ' ----------------------------------------------------- free AMmedia
        DsUtils.FreeAMMediaType(AMmedia)

        ' ----------------------------------------------------- 
        Capture_ConnectFiltersAndRun()
    End Sub


    Friend Function Capture_GetValidVideoFormats() As String()
        '
        If CaptureFilter Is Nothing Or VideoCaptureDevice_Is_VFW() Then Return New String(0) {""}
        '
        Dim videoCaps As VideoCapabilities = New VideoCapabilities(Capture_GetIAMStreamConfig)
        '
        Return videoCaps.ValidFormats(VideoFormatParams.VideoSize)
    End Function

    Friend Function Capture_GetValidVideoSizes() As String()
        '
        If CaptureFilter Is Nothing Or VideoCaptureDevice_Is_VFW() Then Return New String(0) {""}
        '
        Dim videoCaps As VideoCapabilities = New VideoCapabilities(Capture_GetIAMStreamConfig)
        '
        Return videoCaps.ValidSizes(VideoFormatParams.VideoFormat)
    End Function

    Friend Function Capture_GetValidVideoFPS() As String()
        '
        If CaptureFilter Is Nothing Or VideoCaptureDevice_Is_VFW() Then Return New String(0) {""}
    
        Dim videoCaps As VideoCapabilities = New VideoCapabilities(Capture_GetIAMStreamConfig)
        '
        Return videoCaps.ValidFPS(VideoFormatParams.VideoFormat, VideoFormatParams.VideoSize)
    End Function

End Module










