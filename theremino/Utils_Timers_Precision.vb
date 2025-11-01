Module Utils_Timers_Precision


    ' ------------------------------------------------
    '  Usage (simple - ready to copy and paste)
    ' ------------------------------------------------
    ' Dim t As PrecisionTimer = New PrecisionTimer
    ' t.DebugPrintMicrosec()
    ' ------------------------------------------------



    ' ------------------------------------------------
    '   Usage (simple)
    ' ------------------------------------------------
    ' Dim t As PrecisionTimer = New PrecisionTimer      ' <-- implicit start point
    ' ..........
    ' ..........
    ' t.DebugPrintMicrosec()                            ' <-- implicit stop point
    ' ------------------------------------------------



    ' ------------------------------------------------
    '   Usage (start/stop)
    ' ------------------------------------------------
    ' Dim tm1 As PrecisionTimer = New PrecisionTimer 
    ' Dim tm2 As PrecisionTimer = New PrecisionTimer
    ' ..........
    ' tm1.StartTimer                                    ' <-- explicit start point
    ' ..........
    ' tm2.StartTimer                                    ' <-- explicit start point
    ' ..........
    ' tm1.StopTimer                                     ' <-- explicit stop point
    ' ..........
    ' ..........
    ' tm2.StopTimer                                     ' <-- explicit stop point
    ' .........
    ' .........
    ' Debug.Print(tm1.GetTimeMicrosec())                ' <-- implicit stop not used because the tm1 is already stopped
    ' Debug.Print(tm2.GetTimeMicrosec())                ' <-- implicit stop not used because the tm2 is already stopped
    ' ------------------------------------------------


    Public Class PrecisionTimer

        <System.Runtime.InteropServices.DllImport("Kernel32.dll")> _
        Private Shared Function QueryPerformanceFrequency(ByRef countsPerSecondForThisMachine As Int64) As Boolean
        End Function

        <System.Runtime.InteropServices.DllImport("Kernel32.dll")> _
        Private Shared Function QueryPerformanceCounter(ByRef counts As Int64) As Boolean
        End Function

        Private Shared Function QueryPerformanceCounter() As Int64
            Dim counts As Int64
            QueryPerformanceCounter(counts)
            Return counts
        End Function

        Private Shared Function QueryPerformanceFrequency() As Int64
            Dim countsPerSecondForThisMachine As Int64
            QueryPerformanceFrequency(countsPerSecondForThisMachine)
            Return countsPerSecondForThisMachine
        End Function

        Private frequency As Int64
        Private tstart As Int64
        Private tstop As Int64



        ' =================================================================================
        '  Public Timer functions
        ' =================================================================================

        ' ---------------------------------------------- Timer object instancing ( with implicit StartTimer ) 
        Public Sub New()
            frequency = QueryPerformanceFrequency()
            StartTimer()
        End Sub


        ' ---------------------------------------------- Explicit StartTimer and StopTimer
        Public Sub StartTimer()
            tstop = 0
            tstart = QueryPerformanceCounter()
        End Sub

        Public Sub StopTimer()
            If tstop = 0 Then tstop = QueryPerformanceCounter()
        End Sub


        ' ---------------------------------------------- GetTime functions ( with implict StopTimer )
        Public Function GetTime() As Double
            If tstop = 0 Then tstop = QueryPerformanceCounter()
            Return CDbl((tstop - tstart) / frequency)
        End Function

        Public Function GetTimeMillisec() As Double
            If tstop = 0 Then tstop = QueryPerformanceCounter()
            Return Int(1000 * ((tstop - tstart) / frequency))
        End Function

        Public Function GetTimeMicrosec() As Double
            If tstop = 0 Then tstop = QueryPerformanceCounter()
            Return Int(1000000 * ((tstop - tstart) / frequency))
        End Function

        Public Sub SleepToMillisecAndRestartTimer(ByVal millisec As Double)
            Dim TimeToSleep As Double
            TimeToSleep = millisec - 1000 * ((QueryPerformanceCounter() - tstart) / frequency)
            '
            If TimeToSleep > 0 And TimeToSleep <= millisec Then
                System.Threading.Thread.Sleep(CInt(TimeToSleep))
            End If
            '
            tstart = QueryPerformanceCounter()
        End Sub

        Public Function GetFormattedTime(Optional ByVal strFormat As String = "0.000") As String
            Return GetTime.ToString(strFormat)
        End Function

        Public Sub DebugPrintMicrosec()
            If tstop = 0 Then tstop = QueryPerformanceCounter()
            Debug.Print(CStr(Int(1000000 * ((tstop - tstart) / frequency))))
        End Sub

    End Class

End Module
