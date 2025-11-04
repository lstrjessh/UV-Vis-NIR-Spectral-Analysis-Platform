
Friend Class SamplegrabberCallback
    Implements ISampleGrabberCB

    Friend Width As Int32
    Friend Height As Int32

    Friend Function BufferCB(ByVal SampleTime As Double, ByVal pBuffer As System.IntPtr, ByVal BufferLen As Integer) As Integer Implements ISampleGrabberCB.BufferCB
        '
        If BufferLen = 0 Then Return 0
        '
        Dim nBytes As Int32 = Width * Height * 3
        Dim stride As Int32 = Width * 3
        '
        If Not Capture_NewImageIsReady Then
            If BufferLen = nBytes Then
                Capture_Image = New Bitmap(Width, Height, stride, Imaging.PixelFormat.Format24bppRgb, pBuffer)
            End If
            Capture_NewImageIsReady = True
        End If
        '
        ' -------------------------------------------------------
        CalcFramesPerSec(SampleTime)
        Return 0
    End Function

    '
    Private Function SampleCB(ByVal SampleTime As Double, ByVal pSample As IMediaSample) As Integer Implements ISampleGrabberCB.SampleCB
        '
        Dim bufferlen As Integer = pSample.GetActualDataLength
        If bufferlen = 0 Then Return 0
        '
        Dim nBytes As Int32 = Width * Height * 3
        Dim stride As Int32 = Width * 3 '步幅

        If bufferlen = nBytes Then
            Dim pbuffer As IntPtr = Nothing
            pSample.GetPointer(pbuffer)
            Capture_Image = New Bitmap(Width, Height, stride, Imaging.PixelFormat.Format24bppRgb, pbuffer)
        End If
        ' ------------------------------------------------- without this could lock
        Runtime.InteropServices.Marshal.ReleaseComObject(pSample)
        ' -------------------------------------------------
        CalcFramesPerSec(SampleTime)
        Return 0
    End Function


    Private Sub CalcFramesPerSec(ByVal sampletime As Double)
        Static oldSampleTime As Double
        Dim millisec As Double = sampletime - oldSampleTime
        oldSampleTime = sampletime
        If millisec > 0 And millisec < 100000 Then
            Capture_FramesPerSecond += (1 / millisec - Capture_FramesPerSecond) / 4
        End If
    End Sub

End Class
