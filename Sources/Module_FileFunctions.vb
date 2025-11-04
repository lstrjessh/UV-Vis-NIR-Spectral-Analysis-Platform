Module Module_FileFunctions

    ' =======================================================================================================
    '   Files
    ' =======================================================================================================
    ' returns lower-case extension with initial dot
    Public Function GetExtension(ByVal str As String) As String
        On Error Resume Next
        Return LCase(IO.Path.GetExtension(str))
    End Function
    Public Function RemoveExtension(ByVal str As String) As String
        On Error Resume Next
        Return PlatformAdjustedFileName(IO.Path.GetDirectoryName(str) & "\" & IO.Path.GetFileNameWithoutExtension(str))
    End Function
    Public Function FileExists(ByVal fileName As String) As Boolean
        Return My.Computer.FileSystem.FileExists(fileName)
    End Function
    Public Function FolderExists(ByVal FolderName As String) As Boolean
        If FolderName.Length < 2 Then Return False
        FolderName = LCase(FolderName)
        Select Case FolderName
            Case "a:\", "b:\", "c:\", "d:\", "e:\", "f:\", "g:\", "h:\", "i:\", "j:\", "k:\"
                Return True
        End Select
        Return My.Computer.FileSystem.DirectoryExists(FolderName)
    End Function

    'Friend Function GetParentFolder(ByVal folder As String) As String
    '    Dim i As Int32
    '    If Right(folder, 1) = "\" Then folder = Left(folder, Len(folder) - 1)
    '    i = InStrRev(folder, "\")
    '    If i > 1 Then
    '        folder = Left(folder, i)
    '    Else
    '        folder = ""
    '    End If
    '    GetParentFolder = folder
    'End Function

    'Friend Function FindSoundsPath() As String
    '    Dim path As String = Application.StartupPath & "\"
    '    Do
    '        If FolderExists(path & "Sounds") Then
    '            Return path & "Sounds\"
    '        End If
    '        path = GetParentFolder(path)
    '    Loop Until path = ""
    '    Return ""
    'End Function

    'Friend Function File_GetPath(ByVal str As String) As String
    '    Try
    '        Return IO.Path.GetDirectoryName(str) & "\"
    '    Catch
    '        Return str
    '    End Try
    'End Function

    Friend Sub File_Kill(ByVal filename As String)
        If My.Computer.FileSystem.FileExists(filename) Then
            My.Computer.FileSystem.DeleteFile(filename, FileIO.UIOption.OnlyErrorDialogs, FileIO.RecycleOption.DeletePermanently)
        End If
    End Sub

    Friend Function IncreaseFileNameIndex(ByVal filename As String) As String
        If filename = "" Then Return "Noname_01"
        '
        Dim path As String = IO.Path.GetDirectoryName(filename)
        Dim name As String = IO.Path.GetFileNameWithoutExtension(filename)
        Dim ext As String = IO.Path.GetExtension(filename)
        '
        If name = "" Then name = "NoName"
        ' ------------------------------------------------ find index chars
        Dim indexlen As Int32 = 0
        Dim i As Int32 = name.Length - 1
        While Char.IsNumber(name(i)) And i > 0
            i -= 1
            indexlen += 1
        End While
        ' ------------------------------------------------ increase index value
        If indexlen > 0 Then
            Dim number As Int32 = CInt(Val(Mid(name, i + 2)))
            number += 1
            name = Left(name, i + 1) & number.ToString(StrDup(indexlen, "0"))
        Else
            ' -------------------------------------------- or set index to "_01"
            name = name & "_01"
        End If
        ' ------------------------------------------------ compose final file-name
        Return path & "\" & name & ext
    End Function

End Module
