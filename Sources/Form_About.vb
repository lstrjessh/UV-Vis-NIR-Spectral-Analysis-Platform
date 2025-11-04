Public Class Form_About

    Private Sub Form_About_Load(ByVal sender As Object, ByVal e As System.EventArgs) Handles Me.Load
        'Me.Text = AppTitleAndVersion() '直接用定义的名字
    End Sub

    Private Sub Form_About_Shown(ByVal sender As Object, ByVal e As System.EventArgs) Handles Me.Shown
        Label_Info.Text = Msg_About
    End Sub

    Private Sub Button_OK_Click(ByVal sender As System.Object, ByVal e As System.EventArgs) Handles Button_OK.Click
        Close()
    End Sub

    Private Sub Label_www_freetw_Click(ByVal sender As System.Object, ByVal e As System.EventArgs) Handles Label_www_freetw.Click
        Process.Start("http://www.theremino.com")
    End Sub

    Private Sub Label_InfoFreeTW_Click(ByVal sender As System.Object, ByVal e As System.EventArgs) Handles Label_InfoFreeTW.Click
        Send_Mail("engineering@theremino.com", "Info on Theremino System", "Dear Theremino team,\n\n")
    End Sub

    Private Sub Send_Mail(ByVal sendto As String, ByVal subject As String, ByVal body As String)
        subject = subject.Replace(" ", "%20")
        body = body.Replace("\n", "%0d%0a")
        body = body.Replace(" ", "%20")
        Shell("Cmd /C START MAILTO:" & sendto & """?subject=" & subject & "&body=" & body & """")
    End Sub

    Private Sub Labels_MouseEnter(ByVal sender As Object, ByVal e As System.EventArgs) Handles _
                                                                                        Label_www_freetw.MouseEnter, _
                                                                                        Label_InfoFreeTW.MouseEnter
        Dim lb As Label = DirectCast(sender, Label)
        lb.BackColor = Color.FromArgb(250, 240, 255)
    End Sub

    Private Sub Labels_MouseLeave(ByVal sender As Object, ByVal e As System.EventArgs) Handles _
                                                                                        Label_www_freetw.MouseLeave, _
                                                                                        Label_InfoFreeTW.MouseLeave
        Dim lb As Label = DirectCast(sender, Label)
        lb.BackColor = Color.Transparent
    End Sub

End Class