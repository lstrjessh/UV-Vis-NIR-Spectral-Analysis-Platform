
' ==============================================================================================================
'  Operating System and Platform data
' ==============================================================================================================
Module Theremino
    ' ----------------------------------- 
    Friend NAN_Sleep As Single
    Friend NAN_Reset As Single
    Friend NAN_Run As Single
    Friend NAN_Stop As Single
    ' -----------------------------------
    Friend NAN_Recognize As Single
    Friend NAN_ReconnectUSB As Single
    Friend NAN_Calibrate As Single
    '  ----------------------------------
    Friend NAN_MasterError As Single
    Friend NAN_NoMasters As Single
    ' ----------------------------------- 
    Friend OperatingSystemIsWindows As Boolean
    Friend OperatingSystemIs_XP_or_Vista As Boolean
    ' ----------------------------------- 
    Friend Slots As ThereminoSlots = New ThereminoSlots

    Friend Sub InitNanValues()
        ' ------------------------------------------------------- initializing QUIET NAN base values 
        Dim b() As Byte = {&H0, &H0, &HC0, &H7F} ' LSB...MSB
        ' ------------------------------------------------------- QNAN 1 = SLEEP (unpower Servo Motors)(unimplemented in HAL)
        b(0) = 1 : NAN_Sleep = BitConverter.ToSingle(b, 0)
        ' ------------------------------------------------------- QNAN 2 = ZERO  (reset Stepper destination to zero)(unimplemented in HAL)
        b(0) = 2 : NAN_Reset = BitConverter.ToSingle(b, 0)
        ' ------------------------------------------------------- QNAN 3 = RUN   (unimplemented in HAL)
        b(0) = 3 : NAN_Run = BitConverter.ToSingle(b, 0)
        ' ------------------------------------------------------- QNAN 4 = STOP  (unimplemented in HAL)
        b(0) = 4 : NAN_Stop = BitConverter.ToSingle(b, 0)
        ' ------------------------------------------------------- QNAN 5 = RECOGNIZE
        b(0) = 5 : NAN_Recognize = BitConverter.ToSingle(b, 0)
        ' ------------------------------------------------------- QNAN 6 = RECONNECT_USB (unimplemented in HAL)
        b(0) = 6 : NAN_ReconnectUSB = BitConverter.ToSingle(b, 0)
        ' ------------------------------------------------------- QNAN 7 = CALIBRATE
        b(0) = 7 : NAN_Calibrate = BitConverter.ToSingle(b, 0)
        ' ------------------------------------------------------- QNAN 100 = MASTER ERROR (one or more master disconneected)
        b(0) = 100 : NAN_MasterError = BitConverter.ToSingle(b, 0)
        ' ------------------------------------------------------- QNAN 101 = NO MASTERS (no Master has been found by Recognize)
        b(0) = 101 : NAN_NoMasters = BitConverter.ToSingle(b, 0)
    End Sub

    Friend Function IsNanSleep(ByVal n As Single) As Boolean
        If Not Single.IsNaN(n) Then Return False
        Dim b() As Byte = BitConverter.GetBytes(n)
        If b(0) = 1 Then Return True
        Return False
    End Function

    Friend Function IsNanReset(ByVal n As Single) As Boolean
        If Not Single.IsNaN(n) Then Return False
        Dim b() As Byte = BitConverter.GetBytes(n)
        If b(0) = 2 Then Return True
        Return False
    End Function

    Friend Function IsNanRecognize(ByVal n As Single) As Boolean
        If Not Single.IsNaN(n) Then Return False
        Dim b() As Byte = BitConverter.GetBytes(n)
        If b(0) = 5 Then Return True
        Return False
    End Function

    Friend Function IsNanReconnectUSB(ByVal n As Single) As Boolean
        If Not Single.IsNaN(n) Then Return False
        Dim b() As Byte = BitConverter.GetBytes(n)
        If b(0) = 6 Then Return True
        Return False
    End Function

    Friend Function IsNanCalibrate(ByVal n As Single) As Boolean
        If Not Single.IsNaN(n) Then Return False
        Dim b() As Byte = BitConverter.GetBytes(n)
        If b(0) = 7 Then Return True
        Return False
    End Function

    Friend Function NAN_Compare(ByVal n1 As Single, ByVal n2 As Single) As Boolean
        If Not Single.IsNaN(n1) Then Return False
        If Not Single.IsNaN(n2) Then Return False
        Dim b1() As Byte = BitConverter.GetBytes(n1)
        Dim b2() As Byte = BitConverter.GetBytes(n2)
        If b1(0) = b2(0) Then Return True
        Return False
    End Function

    Friend Function NAN_GetName(ByVal n As Single) As String
        If Not Single.IsNaN(n) Then Return ""
        Dim b() As Byte = BitConverter.GetBytes(n)
        Select Case b(0)
            Case 1 : Return "Sleep"
            Case 2 : Return "Reset"
            Case 3 : Return "Run"
            Case 4 : Return "Stop"
            Case 5 : Return "Recognize"
            Case 6 : Return "ReconnectUSB"
            Case 7 : Return "Calibrate"
            Case 100 : Return "Master disconnected ERROR"
            Case 101 : Return "No Masters"
            Case Else : Return "Unknown NAN"
        End Select
    End Function

    Friend Sub InitPlatformData()
        OperatingSystemIsWindows = True
        OperatingSystemIs_XP_or_Vista = False
        Select Case Environment.OSVersion.Platform
            Case PlatformID.Win32NT, _
                 PlatformID.Win32S, _
                 PlatformID.Win32Windows, _
                 PlatformID.WinCE
            Case Else
                OperatingSystemIsWindows = False
        End Select
        If OperatingSystemIsWindows AndAlso Environment.OSVersion.Version.Major <= 6 Then
            OperatingSystemIs_XP_or_Vista = True
        End If
    End Sub

    Friend Function PlatformAdjustedFileName(ByVal FileName As String, _
                                             Optional ByVal DefaultName As String = "") As String
        If OperatingSystemIsWindows Then
            FileName = Replace(FileName, "/", "\")
            Return FileName
        Else
            If FileName.Contains(":") Then FileName = DefaultName
            FileName = Replace(FileName, "\", "/")
            Return FileName
        End If
    End Function

End Module


' ======================================================================================
'  Class ThereminoSlots - Single Class unified for Windows and Unix
' ======================================================================================
Class ThereminoSlots

    Private MMF1 As MemoryMappedFile
    Private MMF1_Unix As MemoryMappedFile_Unix

    Friend Sub New()
        InitNanValues()
        InitPlatformData()
        If OperatingSystemIsWindows Then
            MMF1 = New MemoryMappedFile("Theremino1", 4080)
        Else
            MMF1_Unix = New MemoryMappedFile_Unix
        End If
    End Sub

    Friend Sub New(ByVal SlotFileName As String, Optional ByVal SizeBytes As Int32 = 4080)
        InitNanValues()
        InitPlatformData()
        If OperatingSystemIsWindows Then
            MMF1 = New MemoryMappedFile(SlotFileName, SizeBytes)
        Else
            MMF1_Unix = New MemoryMappedFile_Unix
        End If
    End Sub

    Friend Function ReadSlot(ByVal Slot As Int32) As Single
        If OperatingSystemIsWindows Then
            Return MMF1.ReadSingle(Slot * 4)
        Else
            Return MMF1_Unix.ReadSlot(Slot)
        End If
    End Function

    Friend Function ReadSlot_NoNan(ByVal Slot As Int32) As Single
        Dim n As Single
        If OperatingSystemIsWindows Then
            n = MMF1.ReadSingle(Slot * 4)
        Else
            n = MMF1_Unix.ReadSlot(Slot)
        End If
        If Single.IsNaN(n) Then n = 0
        Return n
    End Function

    Friend Function ReadSlot_NoNan_0_1000(ByVal Slot As Int32) As Single
        Dim n As Single
        If OperatingSystemIsWindows Then
            n = MMF1.ReadSingle(Slot * 4)
        Else
            n = MMF1_Unix.ReadSlot(Slot)
        End If
        If Single.IsNaN(n) Then n = 0
        If n < 0 Then n = 0
        If n > 1000 Then n = 1000
        Return n
    End Function

    Friend Function ReadSlot_Limited_65535(ByVal Slot As Int32) As Int32
        Dim f As Single = MMF1.ReadSingle(Slot * 4)
        If Single.IsNaN(f) Then f = 0
        If f < 0 Then f = 0
        If f > 65535 Then f = 65535
        Return CInt(f)
    End Function

    Friend Sub WriteSlot(ByVal Slot As Int32, ByVal Value As Single)
        If OperatingSystemIsWindows Then
            MMF1.WriteSingle(Slot * 4, Value)
        Else
            MMF1_Unix.WriteSlot(Slot, Value)
        End If
    End Sub

    Friend Sub WriteSlot(ByVal Slot As Int32, ByVal Value As Double)
        If OperatingSystemIsWindows Then
            MMF1.WriteSingle(Slot * 4, CSng(Value))
        Else
            MMF1_Unix.WriteSlot(Slot, CSng(Value))
        End If
    End Sub

    Friend Sub MemoryMappedFile_FillWithNanSleep()
        If OperatingSystemIsWindows Then
            For i As Int32 = 0 To 4080 \ 4 - 1
                MMF1.WriteSingle(i * 4, NAN_Sleep)
            Next
        Else
            For i As Int32 = 0 To 999
                MMF1_Unix.WriteSlot(i, NAN_Sleep)
            Next
        End If
    End Sub


    ' ======================================================================================
    '   CLASS MemoryMappedFile Windows
    ' ======================================================================================
    Private Class MemoryMappedFile

        ' ---------------------------------------------------------------- declararations
        Private Structure SECURITY_ATTRIBUTES
            Const nLength As Int32 = 12
            Public lpSecurityDescriptor As Int32
            Public bInheritHandle As Int32
        End Structure

        Private Declare Function CloseHandle Lib "Kernel32" (ByVal intPtrFileHandle As Int32) As Boolean

        Private Declare Function CreateFileMapping Lib "Kernel32" _
                                                   Alias "CreateFileMappingA" _
                                                   (ByVal hFile As Int32, _
                                                    ByRef lpFileMappigAttributes As SECURITY_ATTRIBUTES, _
                                                    ByVal flProtect As Int32, _
                                                    ByVal dwMaximumSizeHigh As Int32, _
                                                    ByVal dwMaximumSizeLow As Int32, _
                                                    ByVal lpname As String) As Int32

        Private Declare Function MapViewOfFile Lib "Kernel32" _
                                                (ByVal hFileMappingObject As Int32, _
                                                ByVal dwDesiredAccess As Int32, _
                                                ByVal dwFileOffsetHigh As Int32, _
                                                ByVal dwFileOffsetLow As Int32, _
                                                ByVal dwNumberOfBytesToMap As Int32) As IntPtr

        Private Declare Function UnmapViewOfFile Lib "Kernel32" _
                                                (ByVal lpBaseAddress As IntPtr) As Int32

        Private Const PAGE_READWRITE As Int32 = 4
        Private Const FILE_MAP_ALL_ACCESS As Int32 = &H1 Or &H2 Or &H4 Or &H8 Or &H10 Or &HF0000
        Private Const INVALID_HANDLE_VALUE As Int32 = -1

        ' ---------------------------------------------------------------- members
        Private FileHandle As Int32 = 0
        Private MappingAddress As IntPtr = IntPtr.Zero
        Private FileLength As Int32 = 0

        ' ---------------------------------------------------------------- construction / destruction
        Friend Sub New(ByVal Filename As String, Optional ByVal Length As Int32 = 1024)

            FileHandle = CreateFileMapping(INVALID_HANDLE_VALUE, _
                                                 Nothing, _
                                                 PAGE_READWRITE, _
                                                 0, _
                                                 Length, _
                                                 Filename)

            MappingAddress = MapViewOfFile(FileHandle, FILE_MAP_ALL_ACCESS, 0, 0, 0)
            FileLength = Length
        End Sub
        Protected Overrides Sub Finalize()
            Destroy()
            MyBase.Finalize()
        End Sub
        Friend Sub Destroy()
            If MappingAddress <> IntPtr.Zero Then
                UnmapViewOfFile(MappingAddress)
            End If
            If FileHandle <> 0 Then
                CloseHandle(FileHandle)
                FileHandle = 0
            End If
        End Sub

        ' ---------------------------------------------------------------------------- read write
        Friend Sub WriteSingle(ByVal Offset As Int32, ByVal Value As Single)
            If MappingAddress = IntPtr.Zero Then Return
            If Offset < 0 OrElse Offset >= FileLength Then Return
            Dim i As Int32 = BitConverter.ToInt32(BitConverter.GetBytes(Value), 0)
            Runtime.InteropServices.Marshal.WriteInt32(MappingAddress, Offset, i)
        End Sub
        Friend Function ReadSingle(ByVal Offset As Int32) As Single
            If MappingAddress = IntPtr.Zero Then Return 0
            If Offset < 0 OrElse Offset >= FileLength Then Return 0
            Dim i As Int32 = Runtime.InteropServices.Marshal.ReadInt32(MappingAddress, Offset)
            ReadSingle = BitConverter.ToSingle(BitConverter.GetBytes(i), 0)
        End Function

    End Class


    ' ======================================================================================
    '   CLASS MemoryMappedFile_Unix
    ' ======================================================================================
    Private Class MemoryMappedFile_Unix
        ' ---------------------------------------------------------------- declararations

        Private Declare Function the_slot_init Lib "ThereminoSlots" () As Integer
        Private Declare Function the_slot_exit Lib "ThereminoSlots" () As Integer
        Private Declare Sub the_slot_write Lib "ThereminoSlots" (ByVal slot As Integer, ByVal value As Single)
        Private Declare Function the_slot_read Lib "ThereminoSlots" (ByVal slot As Integer) As Single

        ' ---------------------------------------------------------------- construction destruction
        Friend Sub New()
            the_slot_init()
        End Sub
        Friend Sub Destroy()
            the_slot_exit()
        End Sub
        Protected Overrides Sub Finalize()
            Destroy()
            MyBase.Finalize()
        End Sub

        ' ---------------------------------------------------------------- read write
        Friend Sub WriteSlot(ByVal Slot As Int32, ByVal Value As Single)
            the_slot_write(Slot, Value)
        End Sub
        Friend Function ReadSlot(ByVal Slot As Int32) As Single
            Return the_slot_read(Slot)
        End Function

    End Class

End Class
