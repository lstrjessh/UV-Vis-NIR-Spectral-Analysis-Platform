
Imports System.Drawing
Imports System.Drawing.Imaging
Imports System.Drawing.Drawing2D
Imports System.Runtime.InteropServices
Imports Microsoft.VisualBasic.Devices

Module Spectrometer

    Private MeterBar1 As MeterBar = New MeterBar(Form_Main.PictureBox3, 4)
    'm_fillbrush = CreateFillBrush(pbox, 0, 4)  angle=0  blend=4 red

    Friend Sub InitPictureboxImage(ByVal pbox As PictureBox)
        With pbox
            If .ClientSize.Height < 1 Then Return
            If .Image Is Nothing OrElse .Image.Width <> .ClientRectangle.Width OrElse .Image.Height <> .ClientRectangle.Height Then
                .Image = New Bitmap(.ClientRectangle.Width, .ClientRectangle.Height, Imaging.PixelFormat.Format24bppRgb)
            End If
        End With
    End Sub

    Friend TrimPoint1 As Single = 436
    Friend TrimPoint2 As Single = 546

    Friend ReferenceScale As Boolean
    Friend CursorInside As Boolean
    Friend MaxNanometers_OldValue As Int32

    Private ShowDips As Boolean
    Private ShowPeaks As Boolean
    Private UseColors As Boolean
    Private TrimScale As Boolean

    Private SpecV(-1) As Single '����һ�����飬�洢ԭʼ�����ػҶ�ֵ
    Private SpecArray(-1) As Single 'A zero-length array is declared with a dimension of -1.�������У�Spec - spectrum�������ǻ���֮����ֵ
    Private SpecArrayFiltered(-1) As Single '���˺����ֵ
    Private SpecArrayFiltered_IrradianceCoeff(-1) As Single  '������������
    Friend StandardWaveDict As New Dictionary(Of Integer, Single)  '����һ������ϵ�����ֵ�
    Private MaxValue As Single
    Private MaxValueX As Int32

    '�����ӵı��������������������ʾ�����ֵ��ÿ��px��������ɫͨ���ĺ������765��
    '������һЩ����ʱ����˲�ϵ���󣬿��ܻ��һЩ
    Private MaxV As Single = 800.0 'ǰֵΪ780��800 

    Private Filter As Int32
    Private SpeedUP As Int32

    Private SpeedDW As Int32
    Private Flip As Boolean

    '������Щ�������Ǽ�Ȩֵ
    Private KFilter As Single
    Private KSpeedUP As Single
    Private KSpeedDW As Single
    Private Kred As Single
    Private Kgreen As Single
    Private Kblue As Single

    '��Щ����ɫѡ���ĳߴ��λ��
    Private SrcY0 As Int32
    Private SrcDY As Int32
    Private SrcX0 As Int32
    Private SrcDX As Int32

    Private SrcImage As Image     'source ��Դ
    Private SrcBmp As Bitmap
    Private SrcW As Int32
    Private SrcH As Int32
    Private SrcPen As Pen

    Private DestPbox As PictureBox = Form_Main.PBox_Spectrum
    Private DestW As Int32
    Private DestH As Int32
    Private DestRight As Int32
    Private DestBottom As Int32
    Private ScalePen1 As Pen = New Pen(Color.FromArgb(140, 140, 140))
    Private ScalePen2 As Pen = New Pen(Color.FromArgb(200, 200, 200))
    Private ScalePen3 As Pen = New Pen(Color.FromArgb(220, 220, 220))
    Private ScaleFont As Font = New Font("Arial", 8)

    Friend NanometersMin As Single = 270
    Friend NanometersMax As Single = 1200
    Private NanometersDelta As Single  'Delta��������΢�����У���ͨ��������ʾ����

    Private NmStart As Single
    Private NmEnd As Single
    Private NmStartDiv As Int32
    Private NmCoeff As Single 'Coeff��ϵ��

    Private gfx As Graphics
    Private kx As Single
    Private ky As Single
    Private Pen_Graph As Pen = New Pen(Color.Black)

    '�����ֵ� ֻ����һ�ξ���
    Friend Sub MakeIrradianceDict()   '�ȴ���һ������У�����ֵ䣬ֻ���1-�ɹ�����
        '��׼����380-780����401������ÿ��������Ӧ������ϵ������������ֻ�Ǽ�����һ������Ԥ��
        Dim StandardWave() As Integer = Enumerable.Range(380, 401).ToArray()
        '������
        'For i As Integer = 0 To StandardWave.Length - 1
        'Console.WriteLine(StandardWave(i))
        ' Next
        Dim IrradianceCoeff() As Single =
       {
       2.568, 2.568, 2.568, 2.568, 2.568, 2.568, 2.562, 2.532, 2.504, 2.364, 2.282, 2.258, 2.19, 2.072, 1.94, 1.836, 1.753, 1.642, 1.555, 1.49, 1.423, 1.37, 1.3, 1.243, 1.214, 1.18, 1.155, 1.155, 1.16, 1.165, 1.175, 1.188, 1.204, 1.221, 1.25, 1.263, 1.258, 1.277, 1.291, 1.32, 1.348, 1.376, 1.381, 1.376, 1.359, 1.344, 1.326, 1.313, 1.292, 1.273, 1.254, 1.24, 1.218, 1.204, 1.186, 1.173, 1.147, 1.117, 1.089, 1.064, 1.049, 1.04, 1.038, 1.036, 1.036, 1.036, 1.036, 1.038, 1.041, 1.046, 1.049, 1.051, 1.052, 1.055, 1.057, 1.057, 1.056, 1.054, 1.053, 1.049, 1.043, 1.035, 1.028, 1.02, 1.01, 0.996, 0.979, 0.963, 0.945, 0.927, 0.908, 0.893, 0.88, 0.868, 0.857, 0.846, 0.838, 0.828, 0.82, 0.813, 0.81, 0.808, 0.808, 0.809, 0.812, 0.814, 0.816, 0.819, 0.822, 0.827, 0.832, 0.84, 0.85, 0.859, 0.868, 0.874, 0.88, 0.886, 0.891, 0.896, 0.9, 0.904, 0.908, 0.91, 0.909, 0.908, 0.906, 0.903, 0.899, 0.895, 0.891, 0.887, 0.883, 0.878, 0.873, 0.867, 0.86, 0.852, 0.845, 0.839, 0.833, 0.827, 0.822, 0.817, 0.813, 0.808, 0.805, 0.803, 0.802, 0.8, 0.798, 0.796, 0.795, 0.794, 0.794, 0.794, 0.793, 0.793, 0.794, 0.789, 0.786, 0.784, 0.788, 0.793, 0.797, 0.8, 0.804, 0.808, 0.812, 0.815, 0.818, 0.821, 0.824, 0.825, 0.827, 0.827, 0.827, 0.826, 0.824, 0.823, 0.822, 0.822, 0.821, 0.82, 0.818, 0.817, 0.815, 0.813, 0.809, 0.806, 0.803, 0.802, 0.802, 0.801, 0.8, 0.798, 0.796, 0.794, 0.792, 0.791, 0.791, 0.792, 0.795, 0.798, 0.802, 0.806, 0.81, 0.814, 0.82, 0.825, 0.832, 0.839, 0.845, 0.852, 0.858, 0.864, 0.87, 0.877, 0.885, 0.895, 0.905, 0.915, 0.924, 0.935, 0.943, 0.952, 0.958, 0.967, 0.975, 0.984, 0.993, 1.001, 1.012, 1.02, 1.028, 1.031, 1.036, 1.039, 1.042, 1.046, 1.05, 1.055, 1.061, 1.065, 1.069, 1.071, 1.071, 1.069, 1.066, 1.064, 1.062, 1.062, 1.063, 1.064, 1.064, 1.062, 1.058, 1.053, 1.048, 1.044, 1.043, 1.044, 1.045, 1.044, 1.041, 1.036, 1.029, 1.023, 1.018, 1.015, 1.01, 1.007, 1.005, 1.003, 1, 0.995, 0.992, 0.991, 0.99, 0.987, 0.985, 0.984, 0.983, 0.98, 0.978, 0.979, 0.982, 0.985, 0.991, 0.996, 1.002, 1.003, 1.003, 1.001, 1.003, 1.005, 1.008, 1.011, 1.016, 1.018, 1.017, 1.016, 1.018, 1.026, 1.032, 1.039, 1.045, 1.053, 1.057, 1.057, 1.06, 1.064, 1.07, 1.073, 1.076, 1.081, 1.086, 1.087, 1.089, 1.091, 1.093, 1.095, 1.098, 1.102, 1.104, 1.108, 1.111, 1.115, 1.117, 1.118, 1.122, 1.126, 1.128, 1.131, 1.132, 1.134, 1.138, 1.14, 1.142, 1.144, 1.145, 1.146, 1.148, 1.148, 1.145, 1.147, 1.141, 1.138, 1.14, 1.14, 1.14, 1.143, 1.145, 1.149, 1.15, 1.152, 1.156, 1.156, 1.157, 1.161, 1.165, 1.17, 1.172, 1.174, 1.176, 1.179, 1.183, 1.186, 1.189, 1.192, 1.204, 1.2, 1.205, 1.21, 1.215, 1.219, 1.225, 1.23, 1.244, 1.241, 1.246, 1.252, 1.257, 1.264, 1.271, 1.278, 1.283, 1.299, 1.299, 1.309, 1.314, 1.322, 1.332, 1.34, 1.347, 1.367, 1.368, 1.376, 1.387, 1.398, 1.41
        } '770-780


        'Dim StandardWaveDict As New Dictionary(Of Integer, Single)
        For i As Integer = 0 To StandardWave.Length - 1
            If Not StandardWaveDict.ContainsKey(StandardWave(i)) Then
                StandardWaveDict.Add(StandardWave(i), IrradianceCoeff(i))
            End If
        Next
        '������
        'Console.WriteLine("DICTIONARY COUNT: {0}", StandardWaveDict.Count)
    End Sub

    Friend Sub Spectrometer_SetSourceParams()
        If SrcImage Is Nothing Then Return
        ' ---------------------------------------------------------------------
        SrcW = SrcImage.Width '1920����
        SrcH = SrcImage.Height '1080����

        '��ȡFilp��״̬
        Flip = Form_Main.chk_Flip.Checked

        'CInt(n),���ı���������ַ�ת����Int���ͣ�
        '��������ɫѡ����4����ֵ
        Dim StartY As Int32 = Form_Main.txt_StartY.NumericValueInteger
        Dim SizeY As Int32 = Form_Main.txt_SizeY.NumericValueInteger
        Dim StartX As Int32 = Form_Main.txt_StartX.NumericValueInteger
        Dim EndX As Int32 = Form_Main.txt_EndX.NumericValueInteger

        '��ȡ��ɫѡ��򸲸ǵ����ص���ʼλ�úͳߴ磬ʵ��ͼ������Ͻ�Ϊ��0,0������������ͼƬ��ʼ�������꣨SrcX0,SrcY0��
        '��������ͼƬ���ؿ��SrcDX���߶�SrcDY
        SrcX0 = (SrcW * StartX) \ 1000   'StartX-EndX��Χ��1000���൱��һ���ٷֱȣ������ʵ�ʵ����سߴ�
        SrcDX = SrcW - SrcX0 + (SrcW * (EndX - 1000)) \ 1000  '�ó���ȡ���ͼ������س��ȣ�Ҳ������ɫѡ���ĳ���
        SrcDY = (SrcH * SizeY) \ 100    '��ɫѡ���ĸ߶ȣ������ʵ�ʵ����سߴ�
        SrcY0 = SrcH - (SrcH * StartY \ 100) - SrcDY
        ' ---------------------------------------------------------------------
        If SrcX0 + SrcDX > SrcImage.Width Then SrcDX = SrcW - SrcX0
        If SrcDX <= 0 Then SrcX0 += (SrcDX - 1) : SrcDX = 1  '˳��ִ����ð�ŷָ������
        If SrcX0 < 0 Then SrcX0 = 0
        If SrcY0 + SrcDY > SrcH Then SrcDY = SrcH - SrcY0
        If SrcDY <= 0 Then SrcY0 += (SrcDY - 1) : SrcDY = 1
        If SrcY0 < 0 Then SrcY0 = 0
        ' ---------------------------------------------------------------------
        ReDim SpecV(SrcDX - 1) '���������ֶ�Ӧ�����صĸ���
        ReDim SpecArray(SrcDX - 1)  '���¶�������ĳ���
        ReDim SpecArrayFiltered(SrcDX - 1)

        Kred = 1.0F / SrcDY     'ǰֵ��0.5F��ȥ�ݶ���ĳ�1.0F
        Kgreen = 1.0F / SrcDY   'ǰֵ��0.45F��ȥ�ݶ���ĳ�1.0F
        Kblue = 1.0F / SrcDY    'ǰֵ��0.5F��ȥ�ݶ���ĳ�1.0F
        SrcPen = New Pen(Color.FromArgb(200, 120, 0), SrcH \ 100) '������ɫѡ���
        ' ---------------------------------------------------------------------
        Spectrometer_SetScaleTrimParams()
        Spectrometer_ResetReference() 'ȡ���ο����ף�Form_Main.btn_Reference.Checked = False
        'MakeIrradianceDict()
    End Sub

    Friend Sub Spectrometer_SetRunningModeParams()
        ShowDips = Form_Main.btn_Dips.Checked()
        ShowPeaks = Form_Main.btn_Peaks.Checked()
        UseColors = Form_Main.btn_Colors.Checked()
        TrimScale = Form_Main.btn_TrimScale.Checked()
        Filter = Form_Main.txt_Filter.NumericValueInteger
        KFilter = (100 - Filter) / 100.0F + 0.1F
        SpeedUP = Form_Main.txt_RisingSpeed.NumericValueInteger
        KSpeedUP = SpeedUP / 100.0F
        SpeedDW = Form_Main.txt_FallingSpeed.NumericValueInteger
        KSpeedDW = SpeedDW / 100.0F
    End Sub

    ' Version 2.9 - CHANGED from 2000 to 4000
    Friend Sub Spectrometer_SetScaleTrimParams() '���׵�ȫ��Χ�Ĳ������壬Ҳ������������ͼ��Ĳ����ֲ���Χ
        If NanometersMin < 50 Then NanometersMin = 50
        If NanometersMax > 4000 Then NanometersMax = 4000
        If NanometersMax < NanometersMin + 20 Then NanometersMax = NanometersMin + 20
        NanometersDelta = NanometersMax - NanometersMin
        ' ---------------------------------------------------------------------
        NmStart = NanometersMin + NanometersDelta * SrcX0 / SrcW  '���׵���ʼ��������ͼ��ɼ�����ҹ���
        NmEnd = NanometersMax - NanometersDelta * (1.0F - CSng(SrcX0 + SrcDX - 1) / SrcW)
        ' ---------------------------------------------------------------------
        NmStartDiv = 10 * CInt(NmStart / 10.0F)
        ' ---------------------------------------------------------------------
        SetNmCoeff()
    End Sub

    Private Sub Spectrometer_SetDestParams() '����Ŀ����� DestPbox As PictureBox = Form_Main.PBox_Spectrum, dest - destination
        DestW = DestPbox.Image.Width         'DestPbox As PictureBox = Form_Main.PBox_Spectrum
        DestH = DestPbox.Image.Height
        DestRight = DestW - 1
        DestBottom = DestH - 2
        SetNmCoeff()
    End Sub

    Private Sub SetNmCoeff()
        If NmEnd > NmStart Then
            NmCoeff = DestW / (NmEnd - NmStart) '���׵�����ϵ������DestW�벨���ҹ�
        Else
            NmCoeff = 1
        End If
    End Sub

    ' ================================================================================
    '  REFERENCE - SET RESET
    ' ================================================================================
    Private Reference(-1) As Single
    Friend Sub Spectrometer_SetReference()
        Reference = CType(SpecArrayFiltered.Clone, Single())
        For i As Int32 = 0 To SrcDX - 1
            'If Reference(i) < 800 Then Reference(i) = 8000
            If Reference(i) < 1 Then Reference(i) = 99999
        Next
        ReferenceScale = True
    End Sub
    Friend Sub Spectrometer_ResetReference()
        Form_Main.btn_Reference.Checked = False
        ReDim Reference(-1)
        ReferenceScale = False
    End Sub

    ' ================================================================================
    '  PROCESS CAPTURED IMAGE
    ' ================================================================================
    Friend Sub ProcessCapturedImage()
        ' -----------------------------------------------------------------------
        If Not Capture_NewImageIsReady Then Return
        Dim t As PrecisionTimer = New PrecisionTimer
        ' ----------------------------------------------------------------------- LOAD CAPTURED IMAGE
        SrcImage = Capture_Image
        If SrcImage Is Nothing Then Return
        ' ----------------------------------------------------------------------- CAPTURE IMAGE NOT READY
        Capture_NewImageIsReady = False
        ' ----------------------------------------------------------------------- FLIP
        If Flip Then
            SrcImage.RotateFlip(RotateFlipType.RotateNoneFlipX)
        Else
            'SrcImage.RotateFlip(RotateFlipType.RotateNoneFlipNone)
            'SrcImage.RotateFlip(RotateFlipType.RotateNoneFlipY)
        End If
        ' ----------------------------------------------------------------------- PREPARE BITMAP
        SrcBmp = CType(SrcImage, Bitmap)
        ' ----------------------------------------------------------------------- SET PARAMS
        If SrcImage.Width <> SrcW Or SrcImage.Height <> SrcH Then
            Spectrometer_SetSourceParams()
        End If

        '��������ؼ��ĺ�����EXTRACT SPECTRUM
        BitmapToSpectrum()

        'SHOW AREA��������ɫѡ���
        Dim SrcGraphics As Graphics = Graphics.FromImage(SrcBmp)
        SrcGraphics.DrawRectangle(SrcPen, SrcX0, SrcY0, SrcDX, SrcDY) 'SrcPen = New Pen(Color.FromArgb(200, 120, 0), SrcH \ 100)

        ' SHOW SOURCE IMAGE
        Form_Main.PBox_Camera.Image = SrcImage

        'IMAGE INFO��ͼ��ĳߴ��֡��
        Form_Main.Label_Resolution.Text = Capture_Image.Width.ToString & " x " & Capture_Image.Height.ToString
        Form_Main.Label_FramesPerSec.Text = Capture_FramesPerSecond.ToString("0") & " fps"
        Form_Main.Label_Millisec.Text = (t.GetTimeMicrosec / 1000.0F).ToString("0") & " mS"
        ' -----------------------------------------------------------------------
        ShowSpectrumGraph()
        ' ----------------------------------------------------------------------- 
        MeterBar1.SetValue(MaxValue / 256.0F)
        ' ���½ǵ�״̬���� STATUS BAR INDICATIONS
        If Not CursorInside Then  '��겻�ڹ��������ڣ���ʾ������
            If MaxValue > 80 Then 'ǰֵ0.11
                Dim nm As Int32 = CInt(X_To_Nanometers((MaxValueX * DestW) \ SrcDX)) 'NmStart + (x / NmCoeff)��NmCoeff = DestW / (NmEnd - NmStart)
                If nm <> MaxNanometers_OldValue Then
                    MaxNanometers_OldValue = nm
                    Form_Main.Label_MaxPeak.Text = "Max: " & nm.ToString & " nm"
                End If
            Else
                MaxNanometers_OldValue = -1
                Form_Main.Label_MaxPeak.Text = ""
            End If
        End If

    End Sub

    Private Sub BitmapToSpectrum()
        If SrcBmp Is Nothing Then Return
        'Locks a Bitmap into system memory.
        Dim SourceData As BitmapData = SrcBmp.LockBits(New Rectangle(SrcX0, SrcY0, SrcDX, SrcDY), ImageLockMode.ReadWrite, PixelFormat.Format24bppRgb)

        'Gets or sets the stride width (also called scan width) of the Bitmap object.
        Dim SourceStride As Int32 = SourceData.Stride  '����Ӧ����  SrcDX��Strideÿ���ľ���

        'Gets or sets the pixel height of the Bitmap object. Also sometimes referred to as the number of scan lines.
        Dim byteCount As Integer = (SourceData.Stride * SourceData.Height) '�������е���������
        Dim bmpBytes(byteCount - 1) As Byte
        Try
            'Scan0��Gets Or sets the address of the first pixel data in the bitmap. This can also be thought of as the first scan line in the bitmap.
            Marshal.Copy(SourceData.Scan0, bmpBytes, 0, byteCount)  '���Ƶ��������bmpBytes[]�����������һ����ά����
        Catch
        End Try
        SrcBmp.UnlockBits(SourceData)

        Dim sumr As Int32 = 0
        Dim sumg As Int32 = 0
        Dim sumb As Int32 = 0
        Dim disp As Integer 'display
        Dim v As Single
        For x As Int32 = 0 To SrcDX - 1  '���ÿ�����أ�������Y��ֵ
            sumr = 0                    'ÿ��pixel��ÿ����ɫͨ����ȡֵ��Χ��0 - 255��Ҳ����2^8=256
            sumg = 0                    '
            sumb = 0
            disp = x * 3  'ÿһ��������ռ����3���ֽڣ�������Ҫÿ��3���ֽ�ȡһ������
            For y As Int32 = 0 To SrcDY - 1
                sumr += bmpBytes(disp + 2)
                sumg += bmpBytes(disp + 1)
                sumb += bmpBytes(disp)      'opencv ��ͼ��Ĵ洢Ϊ BGR ��ʽ���պú��������е� RGB ��������
                disp = disp + SourceStride  '����ط����ƻ����ˣ�������һ����
            Next

            '�൱�ڶ�ÿһ�����ص�������ɫͨ����ǿ�Ƚ�����ͣ�Kred = 1.0F / SrcDY�����ϵ���Ѿ�����SrcDY��
            v = sumr * Kred + sumg * Kgreen + sumb * Kblue      '����v��ȡֵ��Χ����0-765��255x3 = 765

            SpecV(x) = v

            If v > SpecArray(x) Then   '��Ҫ��������������SpecArray(x)���������
                SpecArray(x) += (v - SpecArray(x)) * KSpeedUP 'txt_RisingSpeed KSpeedUP = SpeedUP / 100.0F���൱��ȡpixelǰ��������ֵ��ƽ��ֵ
                '---------------------------------------------�������ķ�Χ���ǰ������ֵ765����
            Else
                SpecArray(x) += (v - SpecArray(x)) * KSpeedDW
            End If
        Next
        AddFilter()
        AddReference()

    End Sub

    Private Sub AddFilter() 'Noise filtering of the graph.
        MaxValue = 80
        'MaxValue = 0.1
        Dim v, vnew As Single 'v - Value
        For i As Int32 = 0 To SrcDX - 1
            vnew = SpecArray(i)
            ' ----------------------------------------- filter��KFilter = (100 - Filter) / 100.0F + 0.1F
            v += (vnew - v) * KFilter
            ' ----------------------------------------- store filtered value
            SpecArrayFiltered(i) = v    '�������ķ�Χ���ǰ������ֵ765���㣬�����ʵ�����800��
        Next
        For i As Int32 = SrcDX - 1 To 0 Step -1
            vnew = SpecArray(i)
            ' ----------------------------------------- filter
            v += (vnew - v) * KFilter
            ' ----------------------------------------- add up and down filter passes
            'SpecArrayFiltered(i) += v
            SpecArrayFiltered(i) = v
            ' ----------------------------------------- update Max
            If SpecArrayFiltered(i) > MaxValue Then '������ط���ȡ��ֵ�Լ���ֵ��������ʾ����������½�
                'MaxValue = SpecArrayFiltered_IrradianceCoeff(i)  'ǰֵSpecArrayFiltered(i)
                MaxValue = SpecArrayFiltered(i)
                MaxValueX = i
            End If
        Next
    End Sub

    Private Sub AddReference() '���òο����ף����ǲ����չ��׵�ʱ���õ�
        If Reference.Length = 0 Then Return
        Dim v As Single
        For i As Int32 = 0 To SrcDX - 1
            v = SpecArrayFiltered(i)
            ' ----------------------------------------- reference
            v = v * 0.9F * MaxValue / Reference(i)
            If v > MaxValue Then v = MaxValue
            ' ----------------------------------------- store value
            SpecArrayFiltered(i) = v
        Next
    End Sub


    Friend Sub ShowSpectrumGraph()
        ' --------------------------------------------------------------------
        InitPictureboxImage(DestPbox)  'Private DestPbox As PictureBox = Form_Main.PBox_Spectrum
        If DestPbox.Image Is Nothing Then Return
        If DestPbox.Image.Width <> DestW Or DestPbox.Image.Height <> DestH Then
            Spectrometer_SetDestParams()
            gfx = Graphics.FromImage(DestPbox.Image)   'Creates a new Graphics from the specified Image.
        End If
        ' --------------------------------------------------------------------
        'gfx.Clear(Color.Pink)
        '���׻�������ı�����ɫ
        gfx.Clear(Color.AliceBlue) 'Clears the entire drawing surface and fills it with the specified background color.
        Dim x As Single
        Dim y As Single
        Dim drawText As Boolean
        ' --------------------------------------------------------------------- scale X��X������ߣ�10��һ��
        For i As Int32 = NmStartDiv To CInt(NmEnd) Step 10
            'NmStartDiv = 10 * CInt(NmStart / 10.0F)��ת����10�ı����ˣ����ǻ��ƹ��׵���ʼ�����ˣ�Ϊ�����滭������
            drawText = False
            x = (i - NmStart) * NmCoeff
            'NmCoeff = DestW / (NmEnd - NmStart),��Ϊ�˸���ʾ����ҹ�DestPbox���൱������ʾ�����λ�ñ���
            '��Ϊ��Ҫ��DestPbox���滭ͼ

            If i Mod 100 = 0 Then
                gfx.DrawLine(ScalePen1, x, 15, x, DestBottom)   'ScalePen1 = New Pen(Color.FromArgb(140, 140, 140))
                'Draws a line connecting the two points specified by the coordinate pairs.
                'ÿ100nm��һ������
                'DrawLine (pen As Pen, x1 As Integer, y1 As Integer, x2 As Integer, y2 As Integer)

                drawText = True
            ElseIf i Mod 50 = 0 Then
                gfx.DrawLine(ScalePen2, x, 15, x, DestBottom)   'ScalePen2 = New Pen(Color.FromArgb(200, 200, 200))
                If NmCoeff > 2 Then drawText = True
            Else
                gfx.DrawLine(ScalePen3, x, 15, x, DestBottom)   'ScalePen3 = New Pen(Color.FromArgb(220, 220, 220))
                If NmCoeff > 5 Then drawText = True
            End If
            If drawText Then
                gfx.DrawString(i.ToString, ScaleFont, Brushes.Black, x - 4, 1)  'ScaleFont = New Font("Arial", 8)
            End If
        Next
        ' --------------------------------------------------------------------- scale Y
        If ReferenceScale Then
            For i As Int32 = 0 To 110 Step 5
                y = DestBottom - i / 110.0F * (DestBottom - 15)
                If i = 100 Then
                    gfx.DrawLine(Pens.Orange, 0, y, DestRight, y)
                ElseIf i = 50 Then
                    gfx.DrawLine(Pens.Orange, 0, y, DestRight, y)
                ElseIf i Mod 10 = 0 Then
                    gfx.DrawLine(ScalePen2, 0, y, DestRight, y)
                Else
                    gfx.DrawLine(ScalePen3, 0, y, DestRight, y)
                End If
            Next
        Else
            For i As Int32 = 0 To 100 Step 5
                y = DestBottom - i / 100.0F * (DestBottom - 15)
                If i = 100 Then
                    gfx.DrawLine(Pens.YellowGreen, 0, y, DestRight, y)
                ElseIf i = 50 Then
                    gfx.DrawLine(Pens.YellowGreen, 0, y, DestRight, y)
                ElseIf i Mod 10 = 0 Then
                    gfx.DrawLine(ScalePen2, 0, y, DestRight, y)
                Else
                    gfx.DrawLine(ScalePen3, 0, y, DestRight, y)
                End If
            Next
        End If
        '����Ĳ����ǻ�������

        ' --------------------------------------------------------------------- graph vars
        Dim oldx As Single = -99
        Dim oldy As Single = 0
        'ÿһ�����ط����X�᳤�ȣ�DestW = DestPbox.Image.Width, DestRight = DestW - 1, DestBottom = DestH - 2
        kx = CSng(DestRight) / (SpecArrayFiltered.Length - 1)

        'ÿһ�����ǿ�ȷ����Y��߶�,�޸ĵĻ��������ֵ��Ϊ255������ǿ�ȵķ�Χ���Ǵ�0-255����Ȼ����Ҫ��ȥ��������
        'DestBottom - 15�����15��Ҫ����ֵ�ͳ���������
        ky = (DestBottom - 15) / MaxV   'ǰֵ��MaxValue,MaxV��֮ǰ��800�ĵ�780��������ֵ�ĸ߶����õ���


        ' --------------------------------------------------------------------- graph color fill
        '�������������ϵ��
        Dim UV380 As Single = 2.568  '����С��410nm�ģ�һ�ɳ������ϵ��
        Dim IR780 As Single = 1.41  '��������700nm�ģ����ϵ��������Ϊ���Ժ�����ⲿ��
        Dim AverageBGNoise As Single = 2.2  '����ƽ���������� ʵ��Ӧ��Ϊ2.2
        'Dim SpecArrayFiltered(-1) As Single
        ReDim SpecArrayFiltered_IrradianceCoeff(SpecArrayFiltered.Length - 1)    '���¶��峤��
        'SpecArrayFiltered.CopyTo(SpecArrayFiltered_IrradianceCoeff, 0)  '���ø������飬����ֱ�Ӹ�ֵ

        Dim xnew As Int32
        Dim NmWavelength As Int32
        Dim xold As Int32
        Dim x3 As Int32
        Dim y1 As Single
        Dim y2 As Single

        '��SpecArrayFiltered_IrradianceCoeff���鸳ֵ
        For i As Int32 = 0 To SpecArrayFiltered.Length - 1
            NmWavelength = CInt(X_To_Nanometers(BinToX(i)))   'ת���ɲ���,����

            'ͨ�������ж���Ҫ���ԵĶ�Ӧ�ķ���У��ϵ��
            '�������������ķ�ʽ�����嶼����SpecArrayFiltered(i) - AverageBGNoise��Ȼ���ٳ��Զ�Ӧ������ϵ��
            If NmWavelength < 380 Then
                If SpecArrayFiltered(i) - AverageBGNoise >= 0 Then
                    SpecArrayFiltered_IrradianceCoeff(i) = (SpecArrayFiltered(i) - AverageBGNoise) * UV380
                    If SpecArrayFiltered_IrradianceCoeff(i) > MaxV Then
                        SpecArrayFiltered_IrradianceCoeff(i) = MaxV
                    End If
                Else
                    SpecArrayFiltered_IrradianceCoeff(i) = 0
                End If
            ElseIf NmWavelength < 781 Then
                If SpecArrayFiltered(i) - AverageBGNoise >= 0 Then
                    SpecArrayFiltered_IrradianceCoeff(i) = (SpecArrayFiltered(i) - AverageBGNoise) * StandardWaveDict(NmWavelength)
                    If SpecArrayFiltered_IrradianceCoeff(i) > MaxV Then
                        SpecArrayFiltered_IrradianceCoeff(i) = MaxV
                    End If
                Else
                    SpecArrayFiltered_IrradianceCoeff(i) = 0
                End If
            Else
                If SpecArrayFiltered(i) - AverageBGNoise >= 0 Then
                    SpecArrayFiltered_IrradianceCoeff(i) = SpecArrayFiltered(i) * IR780
                    If SpecArrayFiltered_IrradianceCoeff(i) > MaxV Then
                        SpecArrayFiltered_IrradianceCoeff(i) = MaxV
                    End If
                Else
                    SpecArrayFiltered_IrradianceCoeff(i) = 0
                End If
            End If
        Next

        If UseColors Then
            For i As Int32 = 0 To SpecArrayFiltered.Length - 1
                xnew = CInt(BinToX(i))                                              'i * kx������ת����DestPbox������

                If xnew = xold + 1 Then                                              '1nm��Ӧ1�����ص�
                    y = SpecArrayFiltered_IrradianceCoeff(i) * ky                    'y���ǻ�ͼʱ�������ֵ������ǿ�ȣ���0-��DestBottom - 15��,ǰֵSpecArrayFiltered(CInt(i)) * ky
                    If y > 2 Then                                                    '������һ���ǿ�Ⱦ��㣬�Ϳ�ʼ����
                        Pen_Graph.Color = WavelengthToColor(X_To_Nanometers(xnew))  'X_To_Nanometers = NmStart + (xnew / NmCoeff), NmCoeff = DestW / (NmEnd - NmStart)
                        gfx.DrawLine(Pen_Graph, xnew, DestBottom, xnew, DestBottom - y)
                    End If
                ElseIf xnew > xold Then
                    Pen_Graph.Color = WavelengthToColor(X_To_Nanometers(xnew))
                    y1 = SpecArrayFiltered_IrradianceCoeff(CInt(i - 1)) * ky
                    y2 = SpecArrayFiltered_IrradianceCoeff(CInt(i)) * ky
                    If y1 > 2 Or y2 > 2 Then
                        For x3 = xold + 1 To xnew                               '����ط���Ϊ��ÿnm����һ������ǿ��ֵ��������Ϊ�˸�������
                            y = y1 + (y2 - y1) * (x3 - xold) / (xnew - xold)
                            gfx.DrawLine(Pen_Graph, x3, DestBottom, x3, DestBottom - y)
                        Next
                    End If
                End If
                xold = xnew
            Next
        End If
        'graph black line�� ���ײ�ɫ���򶥲��ĺ���
        Pen_Graph.Color = Color.FromArgb(0, 140, 0)
        For i As Int32 = 0 To SpecArrayFiltered.Length - 1
            x = BinToX(i)
            y = DestBottom - SpecArrayFiltered_IrradianceCoeff(i) * ky
            gfx.DrawLine(Pen_Graph, oldx, oldy, x, y)
            'gfx.DrawLine(Pen_Graph, oldx + 1, oldy, x + 1, y) ' thick line
            If NmCoeff > 10 Then
                gfx.DrawRectangle(Pen_Graph, x - 1, y - 1, 3, 3) ' points
            End If
            oldx = x
            oldy = y
        Next
        ' --------------------------------------------------------------------- 
        If TrimScale Then
            MarkTrimPoint(TrimPoint1)
            MarkTrimPoint(TrimPoint2)
        End If
        MarkAllPeaks()
        ' --------------------------------------------------------------------- 
        DestPbox.Invalidate()
    End Sub

    Friend Function X_To_Nanometers(ByVal x As Single) As Single
        If NmCoeff = 0 Then Return 0
        Return NmStart + (x / NmCoeff)
    End Function

    Friend Function X_From_Nanometers(ByVal nm As Single) As Single
        Return (nm - NmStart) * NmCoeff
    End Function

    Friend Function BinToX(ByVal bin As Int32) As Single
        Return bin * kx
    End Function

    Friend Function XtoBin(ByVal x As Single) As Int32
        If kx = 0 Then Return 0
        Return CInt(x / kx)
    End Function


    Private Font_Peaks As Font = New Font("Arial", 9)
    Private Pen_Trim1 As Pen = New Pen(Color.White)

    Private Sub MarkTrimPoint(ByVal nm As Single)   '��ɫ�������ɫ
        Pen_Trim1.DashStyle = DashStyle.Dot
        Dim x As Single
        Dim w As Int32 = 26
        x = X_From_Nanometers(nm)               'Single  Return (nm - NmStart) * NmCoeff
        If x >= 0 And x < DestW Then
            Dim s As String = nm.ToString("0")
            If s.Length > 3 Then w = 34
            gfx.DrawLine(Pens.Black, x, 16, x, DestH)
            gfx.DrawLine(Pen_Trim1, x, 16, x, DestH)
            gfx.FillRectangle(Brushes.Yellow, x - 14, 0, w, 14)
            gfx.DrawRectangle(Pens.Red, x - 14, 0, w, 14)
            gfx.DrawString(s, Font_Peaks, Brushes.Black, x - 13, 0)
        End If
    End Sub

    Private Sub MarkPeak(ByVal bin As Int32, ByVal IsPeak As Boolean)  '��ʾ��ֵ�ͷ��
        Dim x As Single
        Dim y1 As Int32
        Dim y2 As Int32
        Dim w As Int32 = 26
        If bin > 0 Then
            x = BinToX(bin)
            If x >= 0 And x < DestW Then
                Dim s As String = X_To_Nanometers(x).ToString("0")
                If s.Length > 3 Then w = 34
                y1 = 15 + CInt((DestH - 15) * (1 - SpecArrayFiltered_IrradianceCoeff(bin) / MaxV)) 'ǰֵ��MaxValue    SpecArrayFiltered(bin)
                If IsPeak Then                                      '��MarkAllPeaks()���ã���ֵ�����������
                    y2 = y1 - 20                                    '������ʾ��λ��
                    If y2 < DestH - 50 Then y2 = DestH - 20
                    gfx.DrawLine(Pens.Red, x, y1 + 1, x, DestH)
                Else                                                 '��MarkAllPeaks()���ã���ȵ����������
                    y2 = 30                                          '������ʾ��λ��
                    gfx.DrawLine(Pens.Green, x, 40, x, y1 - 3)
                End If
                gfx.FillRectangle(Brushes.Yellow, x - 14, y2, w, 14)
                gfx.DrawRectangle(Pens.Green, x - 14, y2, w, 14)
                gfx.DrawString(s, Font_Peaks, Brushes.Black, x - 13, y2) 'Private Font_Peaks As Font = New Font("Arial", 9)
            End If
        End If
    End Sub

    Private Sub MarkAllPeaks()
        Dim delta As Int32 = (20 * SrcDX) \ DestW
        If delta < 2 Then delta = 2
        Dim v As Single
        Dim valid As Boolean
        For i As Int32 = delta To SpecArrayFiltered.Length - delta - 1
            v = SpecArrayFiltered(i)
            If ShowPeaks Then
                If v > SpecArrayFiltered(i + 1) AndAlso v > SpecArrayFiltered(i - 1) AndAlso v * 100 > MaxValue Then
                    valid = True
                    For d As Int32 = 2 To delta
                        If v < SpecArrayFiltered(i + d) OrElse v < SpecArrayFiltered(i - d) Then
                            valid = False
                            Exit For
                        End If
                    Next
                    If valid Then MarkPeak(i, True)
                End If
            End If
            If ShowDips Then
                If v < SpecArrayFiltered(i + 1) AndAlso v < SpecArrayFiltered(i - 1) AndAlso v * 10000000 > MaxValue Then
                    valid = True
                    For d As Int32 = 2 To delta
                        If v > SpecArrayFiltered(i + d) OrElse v > SpecArrayFiltered(i - d) Then
                            valid = False
                            Exit For
                        End If
                    Next
                    If valid Then MarkPeak(i, False)
                End If
            End If
        Next
    End Sub

    ' ======================================================================================
    '  SPECTRUM FILE
    ' ======================================================================================
    Friend SpectrumFileSeparator As String = vbTab

    Friend Function GetSpectrumText() As String   '�������ݱ�
        Dim GCI As Globalization.CultureInfo = Globalization.CultureInfo.InvariantCulture
        Dim s As String = ""
        If SpectrumFileSeparator = vbTab Then
            s += "Nanometers" + vbTab + "Intensity(a.u.)" + vbCrLf     '+ vbTab + "Counts"
            For i As Int32 = 0 To SpecArrayFiltered.Length - 1
                s += X_To_Nanometers(BinToX(i)).ToString("0.0", GCI) + vbTab + CalcPercentual(SpecArrayFiltered_IrradianceCoeff(i)).ToString("0.0", GCI) + vbCrLf
                '+ vbTab + CalcPercentual(SpecArrayFiltered(i)).ToString("0.0", GCI)
            Next
        Else
            's += "Nanometers   Percentual" + vbCrLf
            's += "-----------------------" + vbCrLf
            'For i As Int32 = 0 To SpecArrayFiltered.Length - 1
            '    s += X_To_Nanometers(BinToX(i)).ToString("0.0", GCI) + SpectrumFileSeparator + CalcPercentual(SpecArrayFiltered(i)).ToString("0.0", GCI).PadLeft(12) + vbCrLf
            'Next
        End If
        Return s
    End Function

    Private Function CalcPercentual(ByVal value As Single) As Single
        Dim perc As Single
        If ReferenceScale Then
            perc = value * 110 / MaxValue
        Else
            'perc = value * 100 / MaxValue
            perc = value * 100 / 100
        End If
        Return perc
    End Function


End Module
