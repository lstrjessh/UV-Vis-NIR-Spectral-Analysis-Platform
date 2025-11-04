Imports System.Drawing.Drawing2D

Public Class MeterBar    '米尺，这个Class指的是右下角的光照强度指示条！

    Private WithEvents pbox As PictureBox
    Private m_graphics As Graphics
    Private m_fillbrush As LinearGradientBrush
    Private m_backbrush As Brush
    Private m_colors As Int32
    Private m_value As Single
    Private w As Int32
    Private h As Int32
    Private Shared m_textfontV As Font = New Font("Arial", 12)
    Private Shared m_textfontH As Font = New Font("Arial", 12)

    Friend Sub New(ByRef p As PictureBox, ByVal colors As Int32)
        If p Is Nothing Then Return
        pbox = p
        pbox.BackColor = Color.Beige   '米色，淡棕色； Color.FromArgb(60, 70, 80)
        m_colors = colors
        SetAspect()
    End Sub

    Friend Sub SetAspect()   '设置外观、横纵比
        InitPictureboxImage(pbox)
        If pbox.Image Is Nothing Then Return
        w = pbox.Image.Width
        h = pbox.Image.Height
        m_graphics = Graphics.FromImage(pbox.Image)
        m_backbrush = New SolidBrush(pbox.BackColor)
        m_graphics.FillRectangle(m_backbrush, 0, 0, w, h)
        Select Case m_colors
            Case 1 : m_fillbrush = CreateFillBrush(pbox, 90, 1)
            Case 2 : m_fillbrush = CreateFillBrush(pbox, 90, 2)
            Case 3 : m_fillbrush = CreateFillBrush(pbox, 0, 3)
            Case 4 : m_fillbrush = CreateFillBrush(pbox, 0, 4)
            Case 5 : m_fillbrush = CreateFillBrush(pbox, 0, 5)
        End Select
    End Sub

    Friend Sub ShowValue()
        If m_graphics Is Nothing Then Return
        If m_fillbrush Is Nothing Then Return
        Dim y As Single = (h - m_value) + 1
        m_graphics.FillRectangle(m_backbrush, 0, 0, w, y - 1)
        m_graphics.FillRectangle(m_fillbrush, 0, y, w, h)
        pbox.Invalidate()
    End Sub

    Friend Sub SetValue(ByVal value As Single)   'value需要 Spectrometer 来计算得出
        If value < 0 Then value = 0
        If value > 1 Then value = 1
        'value = CSng(Math.Sqrt(value))
        value = CInt(value * h)
        If value <> m_value Then
            m_value = value
            ShowValue()
        End If
        If pbox.ClientSize.Width <> w Or pbox.ClientSize.Height <> h Then
            SetAspect()
            ShowValue()
        End If
    End Sub

End Class
