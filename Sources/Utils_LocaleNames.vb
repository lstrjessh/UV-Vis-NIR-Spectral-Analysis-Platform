Imports System.Windows.Forms

Module Utils_LocaleNames

    Friend Language As String = "ENG"
    Private ls(1, -1) As String

    Friend Sub SetLocales()
        ' ------------------------------------- read the file
        ReadLocaleFile()
        ' ------------------------------------- rename
        RenameControls(Form_Main)
        RenameControls(Form_VideoInControls)
        ' ------------------------------------- release the array
        ReDim ls(-1, -1)
    End Sub

    Private Sub ReadLocaleFile()
        Dim locfilename As String = PlatformAdjustedFileName(Application.StartupPath & "\Docs\Language_" & Language & ".txt")
        If Not FileExists(locfilename) Then
            locfilename = PlatformAdjustedFileName(Application.StartupPath & "\Docs\Language_ENG.txt")
        End If
        If FileExists(locfilename) Then
            Dim i As Int32 = -1
            Dim s0 As String
            Dim s1 As String
            '
            Dim f As IO.StreamReader
            f = New System.IO.StreamReader(locfilename, System.Text.Encoding.Default)
            '
            Do While Not f.EndOfStream
                s1 = f.ReadLine
                s1 = Trim(s1.Replace(vbTab, " "))
                s0 = ExtractParamName(s1)
                If s0.Length > 1 And s1.Length > 1 Then
                    If s0.StartsWith("Msg_") Then
                        AddMessage(s0, s1)
                    Else
                        i += 1
                        ' ---------------------------------------------
                        RedimPreserve_2D_Array(ls, 1, i)
                        ' --------------------------------------------- 
                        ls(0, i) = s0
                        ls(1, i) = s1
                    End If
                End If
            Loop
            f.Close()
        End If
    End Sub

    ' ===============================================================================================
    '  RENAME CONTROLS
    ' ===============================================================================================
    Private Sub RenameControls(ByVal container As Control)
        '
        For i As Int32 = 0 To ls.GetLength(1) - 1
            If container.Name = ls(0, i) Then
                container.Text = ls(1, i)
            End If
        Next
        '
        For Each ctrl As Control In container.Controls
            For i As Int32 = 0 To ls.GetLength(1) - 1

                If ctrl.Name = ls(0, i) Then
                    ctrl.Text = ls(1, i)
                End If

                If TypeOf ctrl Is ToolStrip Then
                    Dim ts As ToolStrip = DirectCast(ctrl, ToolStrip)
                    RenameToolStripItems(ts)
                End If

                If TypeOf ctrl Is MenuStrip Then
                    Dim ms As MenuStrip = DirectCast(ctrl, MenuStrip)
                    RenameMenuStripItems(ms)
                End If
            Next
            ' ---------------------------- Recursively call this function for any container controls.
            If container.HasChildren Then
                RenameControls(ctrl)
            End If
        Next
    End Sub

    Private Sub RenameToolStripItems(ByRef ts As ToolStrip)
        For i As Int32 = 0 To ls.GetLength(1) - 1
            For j As Int32 = 0 To ts.Items.Count - 1
                If ts.Items(j).Name = ls(0, i) Then
                    ts.Items(j).Text = " " & ls(1, i)
                End If
            Next
        Next
    End Sub

    Private Sub RenameMenuStripItems(ByRef ms As MenuStrip)
        For i As Int32 = 0 To ls.GetLength(1) - 1
            For j As Int32 = 0 To ms.Items.Count - 1
                If ms.Items(j).Name = ls(0, i) Then
                    ms.Items(j).Text = " " & ls(1, i)
                End If
                Dim tsmi As ToolStripMenuItem = DirectCast(ms.Items(j), ToolStripMenuItem)
                For k As Int32 = 0 To tsmi.DropDownItems.Count - 1
                    If tsmi.DropDownItems(k).Name = ls(0, i) Then
                        tsmi.DropDownItems(k).Text = " " & ls(1, i)
                    End If
                    ' ---------------------------------------------------------- second level menu
                    Dim tsmi2 As ToolStripMenuItem = TryCast(tsmi.DropDownItems(k), ToolStripMenuItem)
                    If tsmi2 IsNot Nothing Then
                        For l As Int32 = 0 To tsmi2.DropDownItems.Count - 1
                            If tsmi2.DropDownItems(l).Name = ls(0, i) Then
                                tsmi2.DropDownItems(l).Text = " " & ls(1, i)
                            End If
                        Next
                    End If
                Next
            Next
        Next
    End Sub


    ' ===============================================================================================
    '  MESSAGES
    ' ===============================================================================================
    Friend Msg_About As String = "Visible, UVA and Near Infrared Spectrometer"
    Friend Msg_TrimTitle As String = "Manual Trim Points"
    Friend Msg_Trim1 As String = "Enter the first trim point in nanometers."
    Friend Msg_Trim2 As String = "Enter the second trim point in nanometers."

    Private Sub AddMessage(ByVal s0 As String, ByVal s1 As String)
        Select Case s0
            Case "Msg_About" : Msg_About = s1
            Case "Msg_TrimTitle" : Msg_TrimTitle = s1
            Case "Msg_Trim1" : Msg_Trim1 = s1
            Case "Msg_Trim2" : Msg_Trim2 = s1
        End Select
    End Sub

End Module
