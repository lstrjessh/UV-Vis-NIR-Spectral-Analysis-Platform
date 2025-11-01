
Imports System.Collections
Imports System.Drawing
Imports System.Runtime.InteropServices
Imports System.Text
Imports System.Reflection
Imports System.Security

Imports System.Runtime.InteropServices.ComTypes

Public Enum PinConnectedStatus
    Unconnected
    Connected
End Enum

<StructLayout(LayoutKind.Sequential)> _
Public Structure BitmapInfo
    Public bmiHeader As BitmapInfoHeader
    Public bmiColors As Integer()
End Structure

<StructLayout(LayoutKind.Sequential, Pack:=2)> _
Public Class BitmapInfoHeader
    Public Size As Integer
    Public Width As Integer
    Public Height As Integer
    Public Planes As Short
    Public BitCount As Short
    Public Compression As Integer
    Public ImageSize As Integer
    Public XPelsPerMeter As Integer
    Public YPelsPerMeter As Integer
    Public ClrUsed As Integer
    Public ClrImportant As Integer
End Class

<StructLayout(LayoutKind.Explicit)> _
Public Structure DDPixelFormat
    <FieldOffset(0)> _
    Public dwSize As Integer
    <FieldOffset(4)> _
    Public dwFlags As Integer
    <FieldOffset(8)> _
    Public dwFourCC As Integer

    <FieldOffset(12)> _
    Public dwRGBBitCount As Integer
    <FieldOffset(12)> _
    Public dwYUVBitCount As Integer
    <FieldOffset(12)> _
    Public dwZBufferBitDepth As Integer
    <FieldOffset(12)> _
    Public dwAlphaBitDepth As Integer

    <FieldOffset(16)> _
    Public dwRBitMask As Integer
    <FieldOffset(16)> _
    Public dwYBitMask As Integer

    <FieldOffset(20)> _
    Public dwGBitMask As Integer
    <FieldOffset(20)> _
    Public dwUBitMask As Integer

    <FieldOffset(24)> _
    Public dwBBitMask As Integer
    <FieldOffset(24)> _
    Public dwVBitMask As Integer

    <FieldOffset(28)> _
    Public dwRGBAlphaBitMask As Integer
    <FieldOffset(28)> _
    Public dwYUVAlphaBitMask As Integer
    <FieldOffset(28)> _
    Public dwRGBZBitMask As Integer
    <FieldOffset(28)> _
    Public dwYUVZBitMask As Integer
End Structure

<StructLayout(LayoutKind.Sequential)> _
Public Structure DsCAUUID
    Public cElems As Integer
    Public pElems As IntPtr

    ' Perform a manual marshaling of pElems to retrieve an array of System.Guid.
    ' Assume this structure has been already filled by the ISpecifyPropertyPages.GetPages() method.
    ' Returns a managed representation of pElems (cElems items)
    Public Function ToGuidArray() As Guid()
        Dim retval As Guid() = New Guid(Me.cElems - 1) {}

        For i As Integer = 0 To Me.cElems - 1
            Dim ptr As New IntPtr(Me.pElems.ToInt64() + (Marshal.SizeOf(GetType(Guid)) * i))
            retval(i) = CType(Marshal.PtrToStructure(ptr, GetType(Guid)), Guid)
        Next

        Return retval
    End Function
End Structure

' DirectShowLib.DsLong is a wrapper class around a <see cref="System.Int64"/> value type.
' This class is necessary to enable null paramters passing.
<StructLayout(LayoutKind.Sequential)> _
Public Class DsLong
    Private Value As Long

    Public Sub New(ByVal Value As Long)
        Me.Value = Value
    End Sub

    ' Get a string representation of this DirectShowLib.DsLong Instance
    Public Overrides Function ToString() As String
        Return Me.Value.ToString()
    End Function

    Public Overrides Function GetHashCode() As Integer
        Return Me.Value.GetHashCode()
    End Function

    Public Shared Widening Operator CType(ByVal l As DsLong) As Long
        Return l.Value
    End Operator

    Public Shared Widening Operator CType(ByVal l As Long) As DsLong
        Return New DsLong(l)
    End Operator

    ' Get the System.Int64 equivalent to this DirectShowLib.DsLong instance
    Public Function ToInt64() As Long
        Return Me.Value
    End Function

    ' Get a new DirectShowLib.DsLong instance for a given System.Int64
    Public Shared Function FromInt64(ByVal l As Long) As DsLong
        Return New DsLong(l)
    End Function
End Class

<StructLayout(LayoutKind.Explicit)> _
Public Class DsGuid
    <FieldOffset(0)> _
    Private guid As Guid

    Public Shared ReadOnly Empty As DsGuid = guid.Empty

    Public Sub New()
        Me.guid = guid.Empty
    End Sub

    Public Sub New(ByVal g As String)
        Me.guid = New Guid(g)
    End Sub

    Public Sub New(ByVal g As Guid)
        Me.guid = g
    End Sub

    Public Overrides Function ToString() As String
        Return Me.guid.ToString()
    End Function

    Public Overloads Function ToString(ByVal format As String) As String
        Return Me.guid.ToString(format)
    End Function

    Public Overrides Function GetHashCode() As Integer
        Return Me.guid.GetHashCode()
    End Function

    Public Shared Widening Operator CType(ByVal g As DsGuid) As Guid
        Return g.guid
    End Operator

    Public Shared Widening Operator CType(ByVal g As Guid) As DsGuid
        Return New DsGuid(g)
    End Operator

    Public Function ToGuid() As Guid
        Return Me.guid
    End Function

    Public Shared Function FromGuid(ByVal g As Guid) As DsGuid
        Return New DsGuid(g)
    End Function
End Class

<StructLayout(LayoutKind.Sequential)> _
Public Class DsInt
    Private Value As Integer

    Public Sub New(ByVal Value As Integer)
        Me.Value = Value
    End Sub

    Public Overrides Function ToString() As String
        Return Me.Value.ToString()
    End Function

    Public Overrides Function GetHashCode() As Integer
        Return Me.Value.GetHashCode()
    End Function

    Public Shared Widening Operator CType(ByVal l As DsInt) As Integer
        Return l.Value
    End Operator

    Public Shared Widening Operator CType(ByVal l As Integer) As DsInt
        Return New DsInt(l)
    End Operator


    Public Function ToInt32() As Integer
        Return Me.Value
    End Function

    Public Shared Function FromInt32(ByVal l As Integer) As DsInt
        Return New DsInt(l)
    End Function
End Class

<StructLayout(LayoutKind.Sequential)> _
Public Class DsShort
    Private Value As Short

    Public Sub New(ByVal Value As Short)
        Me.Value = Value
    End Sub

    Public Overrides Function ToString() As String
        Return Me.Value.ToString()
    End Function

    Public Overrides Function GetHashCode() As Integer
        Return Me.Value.GetHashCode()
    End Function

    Public Shared Widening Operator CType(ByVal l As DsShort) As Short
        Return l.Value
    End Operator

    Public Shared Widening Operator CType(ByVal l As Short) As DsShort
        Return New DsShort(l)
    End Operator

    Public Function ToInt16() As Short
        Return Me.Value
    End Function

    Public Shared Function FromInt16(ByVal l As Short) As DsShort
        Return New DsShort(l)
    End Function
End Class

' A managed representation of the Win32 RECT structure.
<StructLayout(LayoutKind.Sequential)> _
Public Class DsRect
    Public left As Integer
    Public top As Integer
    Public right As Integer
    Public bottom As Integer

    Public Sub New()
        Me.left = 0
        Me.top = 0
        Me.right = 0
        Me.bottom = 0
    End Sub

    Public Sub New(ByVal left As Integer, ByVal top As Integer, ByVal right As Integer, ByVal bottom As Integer)
        Me.left = left
        Me.top = top
        Me.right = right
        Me.bottom = bottom
    End Sub

    Public Sub New(ByVal rectangle As Rectangle)
        Me.left = rectangle.Left
        Me.top = rectangle.Top
        Me.right = rectangle.Right
        Me.bottom = rectangle.Bottom
    End Sub

    Public Overrides Function ToString() As String
        Return String.Format("[{0}, {1} - {2}, {3}]", Me.left, Me.top, Me.right, Me.bottom)
    End Function

    Public Overrides Function GetHashCode() As Integer
        Return Me.left.GetHashCode() Or Me.top.GetHashCode() Or Me.right.GetHashCode() Or Me.bottom.GetHashCode()
    End Function

    Public Overrides Function Equals(ByVal obj As Object) As Boolean
        If TypeOf obj Is DsRect Then
            Dim cmp As DsRect = DirectCast(obj, DsRect)

            Return right = cmp.right AndAlso bottom = cmp.bottom AndAlso left = cmp.left AndAlso top = cmp.top
        End If

        If TypeOf obj Is Rectangle Then
            Dim cmp As Rectangle = CType(obj, Rectangle)

            Return right = cmp.Right AndAlso bottom = cmp.Bottom AndAlso left = cmp.Left AndAlso top = cmp.Top
        End If

        Return False
    End Function

    Public Shared Widening Operator CType(ByVal r As DsRect) As Rectangle
        Return r.ToRectangle()
    End Operator

    Public Shared Widening Operator CType(ByVal r As Rectangle) As DsRect
        Return New DsRect(r)
    End Operator

    Public Function ToRectangle() As Rectangle
        Return New Rectangle(Me.left, Me.top, (Me.right - Me.left), (Me.bottom - Me.top))
    End Function

    Public Shared Function FromRectangle(ByVal r As Rectangle) As DsRect
        Return New DsRect(r)
    End Function
End Class

<StructLayout(LayoutKind.Sequential)> _
Public Structure NormalizedRect
    Public left As Single
    Public top As Single
    Public right As Single
    Public bottom As Single

    Public Sub New(ByVal l As Single, ByVal t As Single, ByVal r As Single, ByVal b As Single)
        Me.left = l
        Me.top = t
        Me.right = r
        Me.bottom = b
    End Sub

    Public Sub New(ByVal r As RectangleF)
        Me.left = r.Left
        Me.top = r.Top
        Me.right = r.Right
        Me.bottom = r.Bottom
    End Sub

    Public Overrides Function ToString() As String
        Return String.Format("[{0}, {1} - {2}, {3}]", Me.left, Me.top, Me.right, Me.bottom)
    End Function

    Public Overrides Function GetHashCode() As Integer
        Return Me.left.GetHashCode() Or Me.top.GetHashCode() Or Me.right.GetHashCode() Or Me.bottom.GetHashCode()
    End Function

    Public Shared Widening Operator CType(ByVal r As NormalizedRect) As RectangleF
        Return r.ToRectangleF()
    End Operator

    Public Shared Widening Operator CType(ByVal r As Rectangle) As NormalizedRect
        Return New NormalizedRect(r)
    End Operator

    Public Shared Operator =(ByVal r1 As NormalizedRect, ByVal r2 As NormalizedRect) As Boolean
        Return ((r1.left = r2.left) AndAlso (r1.top = r2.top) AndAlso (r1.right = r2.right) AndAlso (r1.bottom = r2.bottom))
    End Operator

    Public Shared Operator <>(ByVal r1 As NormalizedRect, ByVal r2 As NormalizedRect) As Boolean
        Return ((r1.left <> r2.left) OrElse (r1.top <> r2.top) OrElse (r1.right <> r2.right) OrElse (r1.bottom <> r2.bottom))
    End Operator

    Public Overrides Function Equals(ByVal obj As Object) As Boolean
        If Not (TypeOf obj Is NormalizedRect) Then
            Return False
        End If
        Dim other As NormalizedRect = CType(obj, NormalizedRect)
        Return (Me = other)
    End Function

    Public Function ToRectangleF() As RectangleF
        Return New RectangleF(Me.left, Me.top, (Me.right - Me.left), (Me.bottom - Me.top))
    End Function

    Public Shared Function FromRectangle(ByVal r As RectangleF) As NormalizedRect
        Return New NormalizedRect(r)
    End Function
End Structure


Public NotInheritable Class DsResults
    Private Sub New()
    End Sub
    Public Const E_InvalidMediaType As Integer = &H80040200
    Public Const E_InvalidSubType As Integer = &H80040201
    Public Const E_NeedOwner As Integer = &H80040202
    Public Const E_EnumOutOfSync As Integer = &H80040203
    Public Const E_AlreadyConnected As Integer = &H80040204
    Public Const E_FilterActive As Integer = &H80040205
    Public Const E_NoTypes As Integer = &H80040206
    Public Const E_NoAcceptableTypes As Integer = &H80040207
    Public Const E_InvalidDirection As Integer = &H80040208
    Public Const E_NotConnected As Integer = &H80040209
    Public Const E_NoAllocator As Integer = &H8004020A
    Public Const E_RunTimeError As Integer = &H8004020B
    Public Const E_BufferNotSet As Integer = &H8004020C
    Public Const E_BufferOverflow As Integer = &H8004020D
    Public Const E_BadAlign As Integer = &H8004020E
    Public Const E_AlreadyCommitted As Integer = &H8004020F
    Public Const E_BuffersOutstanding As Integer = &H80040210
    Public Const E_NotCommitted As Integer = &H80040211
    Public Const E_SizeNotSet As Integer = &H80040212
    Public Const E_NoClock As Integer = &H80040213
    Public Const E_NoSink As Integer = &H80040214
    Public Const E_NoInterface As Integer = &H80040215
    Public Const E_NotFound As Integer = &H80040216
    Public Const E_CannotConnect As Integer = &H80040217
    Public Const E_CannotRender As Integer = &H80040218
    Public Const E_ChangingFormat As Integer = &H80040219
    Public Const E_NoColorKeySet As Integer = &H8004021A
    Public Const E_NotOverlayConnection As Integer = &H8004021B
    Public Const E_NotSampleConnection As Integer = &H8004021C
    Public Const E_PaletteSet As Integer = &H8004021D
    Public Const E_ColorKeySet As Integer = &H8004021E
    Public Const E_NoColorKeyFound As Integer = &H8004021F
    Public Const E_NoPaletteAvailable As Integer = &H80040220
    Public Const E_NoDisplayPalette As Integer = &H80040221
    Public Const E_TooManyColors As Integer = &H80040222
    Public Const E_StateChanged As Integer = &H80040223
    Public Const E_NotStopped As Integer = &H80040224
    Public Const E_NotPaused As Integer = &H80040225
    Public Const E_NotRunning As Integer = &H80040226
    Public Const E_WrongState As Integer = &H80040227
    Public Const E_StartTimeAfterEnd As Integer = &H80040228
    Public Const E_InvalidRect As Integer = &H80040229
    Public Const E_TypeNotAccepted As Integer = &H8004022A
    Public Const E_SampleRejected As Integer = &H8004022B
    Public Const E_SampleRejectedEOS As Integer = &H8004022C
    Public Const E_DuplicateName As Integer = &H8004022D
    Public Const S_DuplicateName As Integer = &H4022D
    Public Const E_Timeout As Integer = &H8004022E
    Public Const E_InvalidFileFormat As Integer = &H8004022F
    Public Const E_EnumOutOfRange As Integer = &H80040230
    Public Const E_CircularGraph As Integer = &H80040231
    Public Const E_NotAllowedToSave As Integer = &H80040232
    Public Const E_TimeAlreadyPassed As Integer = &H80040233
    Public Const E_AlreadyCancelled As Integer = &H80040234
    Public Const E_CorruptGraphFile As Integer = &H80040235
    Public Const E_AdviseAlreadySet As Integer = &H80040236
    Public Const S_StateIntermediate As Integer = &H40237
    Public Const E_NoModexAvailable As Integer = &H80040238
    Public Const E_NoAdviseSet As Integer = &H80040239
    Public Const E_NoFullScreen As Integer = &H8004023A
    Public Const E_InFullScreenMode As Integer = &H8004023B
    Public Const E_UnknownFileType As Integer = &H80040240
    Public Const E_CannotLoadSourceFilter As Integer = &H80040241
    Public Const S_PartialRender As Integer = &H40242
    Public Const E_FileTooShort As Integer = &H80040243
    Public Const E_InvalidFileVersion As Integer = &H80040244
    Public Const S_SomeDataIgnored As Integer = &H40245
    Public Const S_ConnectionsDeferred As Integer = &H40246
    Public Const E_InvalidCLSID As Integer = &H80040247
    Public Const E_InvalidMediaType2 As Integer = &H80040248
    Public Const E_BabKey As Integer = &H800403F2
    Public Const S_NoMoreItems As Integer = &H40103
    Public Const E_SampleTimeNotSet As Integer = &H80040249
    Public Const S_ResourceNotNeeded As Integer = &H40250
    Public Const E_MediaTimeNotSet As Integer = &H80040251
    Public Const E_NoTimeFormatSet As Integer = &H80040252
    Public Const E_MonoAudioHW As Integer = &H80040253
    Public Const S_MediaTypeIgnored As Integer = &H40254
    Public Const E_NoDecompressor As Integer = &H80040255
    Public Const E_NoAudioHardware As Integer = &H80040256
    Public Const S_VideoNotRendered As Integer = &H40257
    Public Const S_AudioNotRendered As Integer = &H40258
    Public Const E_RPZA As Integer = &H80040259
    Public Const S_RPZA As Integer = &H4025A
    Public Const E_ProcessorNotStable As Integer = &H8004025B
    Public Const E_UnsupportedAudio As Integer = &H8004025C
    Public Const E_UnsupportedVideo As Integer = &H8004025D
    Public Const E_MPEGNotConstrained As Integer = &H8004025E
    Public Const E_NotInGraph As Integer = &H8004025F
    Public Const S_Estimated As Integer = &H40260
    Public Const E_NoTimeFormat As Integer = &H80040261
    Public Const E_ReadOnly As Integer = &H80040262
    Public Const S_Reserved As Integer = &H40263
    Public Const E_BufferUnderflow As Integer = &H80040264
    Public Const E_UnsupportedStream As Integer = &H80040265
    Public Const E_NoTransport As Integer = &H80040266
    Public Const S_StreamOff As Integer = &H40267
    Public Const S_CantCue As Integer = &H40268
    Public Const E_BadVideoCD As Integer = &H80040269
    Public Const S_NoStopTime As Integer = &H40270
    Public Const E_OutOfVideoMemory As Integer = &H80040271
    Public Const E_VPNegotiationFailed As Integer = &H80040272
    Public Const E_DDrawCapsNotStable As Integer = &H80040273
    Public Const E_NoVPHardware As Integer = &H80040274
    Public Const E_NoCaptureHardware As Integer = &H80040275
    Public Const E_DVDOperationInhibited As Integer = &H80040276
    Public Const E_DVDInvalidDomain As Integer = &H80040277
    Public Const E_DVDNoButton As Integer = &H80040278
    Public Const E_DVDGraphNotReady As Integer = &H80040279
    Public Const E_DVDRenderFail As Integer = &H8004027A
    Public Const E_DVDDecNotEnough As Integer = &H8004027B
    Public Const E_DDrawVersionNotStable As Integer = &H8004027C
    Public Const E_CopyProtFailed As Integer = &H8004027D
    Public Const S_NoPreviewPin As Integer = &H4027E
    Public Const E_TimeExpired As Integer = &H8004027F
    Public Const S_DVDNonOneSequential As Integer = &H40280
    Public Const E_DVDWrongSpeed As Integer = &H80040281
    Public Const E_DVDMenuDoesNotExist As Integer = &H80040282
    Public Const E_DVDCmdCancelled As Integer = &H80040283
    Public Const E_DVDStateWrongVersion As Integer = &H80040284
    Public Const E_DVDStateCorrupt As Integer = &H80040285
    Public Const E_DVDStateWrongDisc As Integer = &H80040286
    Public Const E_DVDIncompatibleRegion As Integer = &H80040287
    Public Const E_DVDNoAttributes As Integer = &H80040288
    Public Const E_DVDNoGoupPGC As Integer = &H80040289
    Public Const E_DVDLowParentalLevel As Integer = &H8004028A
    Public Const E_DVDNotInKaraokeMode As Integer = &H8004028B
    Public Const S_DVDChannelContentsNotAvailable As Integer = &H4028C
    Public Const S_DVDNotAccurate As Integer = &H4028D
    Public Const E_FrameStepUnsupported As Integer = &H8004028E
    Public Const E_DVDStreamDisabled As Integer = &H8004028F
    Public Const E_DVDTitleUnknown As Integer = &H80040290
    Public Const E_DVDInvalidDisc As Integer = &H80040291
    Public Const E_DVDNoResumeInformation As Integer = &H80040292
    Public Const E_PinAlreadyBlockedOnThisThread As Integer = &H80040293
    Public Const E_PinAlreadyBlocked As Integer = &H80040294
    Public Const E_CertificationFailure As Integer = &H80040295
    Public Const E_VMRNotInMixerMode As Integer = &H80040296
    Public Const E_VMRNoApSupplied As Integer = &H80040297
    Public Const E_VMRNoDeinterlace_HW As Integer = &H80040298
    Public Const E_VMRNoProcAMPHW As Integer = &H80040299
    Public Const E_DVDVMR9IncompatibleDec As Integer = &H8004029A
    Public Const E_NoCOPPHW As Integer = &H8004029B
End Class


Public NotInheritable Class DsError
    Private Sub New()
    End Sub
    <DllImport("quartz.dll", CharSet:=CharSet.Unicode, ExactSpelling:=True, EntryPoint:="AMGetErrorTextW"), SuppressUnmanagedCodeSecurity()> _
    Public Shared Function AMGetErrorText(ByVal hr As Integer, ByVal buf As StringBuilder, ByVal max As Integer) As Integer
    End Function

    Public Shared Sub ThrowExceptionForHR(ByVal hr As Integer)
        ' If a severe error has occurred
        If hr < 0 Then
            Dim s As String = GetErrorText(hr)

            ' If a string is returned, build a com error from it
            If s IsNot Nothing Then
                Throw New COMException(s, hr)
            Else
                ' No string, just use standard com error
                Marshal.ThrowExceptionForHR(hr)
            End If
        End If
    End Sub

    Public Shared Function GetErrorText(ByVal hr As Integer) As String
        Const MAX_ERROR_TEXT_LEN As Integer = 160

        ' Make a buffer to hold the string
        Dim buf As New StringBuilder(MAX_ERROR_TEXT_LEN, MAX_ERROR_TEXT_LEN)

        ' If a string is returned, build a com error from it
        If AMGetErrorText(hr, buf, MAX_ERROR_TEXT_LEN) > 0 Then
            Return buf.ToString()
        End If

        Return Nothing
    End Function
End Class


Public NotInheritable Class DsUtils
    Private Sub New()
    End Sub

    Public Shared Function GetPinCategory(ByVal pPin As IPin) As Guid
        Dim guidRet As Guid = Guid.Empty

        ' Memory to hold the returned guid
        Dim iSize As Integer = Marshal.SizeOf(GetType(Guid))
        Dim ipOut As IntPtr = Marshal.AllocCoTaskMem(iSize)

        Try
            Dim hr As Integer
            Dim cbBytes As Integer
            Dim g As Guid = PropSetID.Pin

            ' Get an IKsPropertySet from the pin
            Dim pKs As IKsPropertySet = TryCast(pPin, IKsPropertySet)

            If pKs IsNot Nothing Then
                ' Query for the Category
                hr = pKs.[Get](g, CInt(AMPropertyPin.Category), IntPtr.Zero, 0, ipOut, iSize, _
                 cbBytes)
                DsError.ThrowExceptionForHR(hr)

                ' Marshal it to the return variable
                guidRet = CType(Marshal.PtrToStructure(ipOut, GetType(Guid)), Guid)
            End If
        Finally
            Marshal.FreeCoTaskMem(ipOut)
            ipOut = IntPtr.Zero
        End Try

        Return guidRet
    End Function

    Public Shared Sub FreeAMMediaType(ByVal mediaType As AMMediaType)
        If mediaType IsNot Nothing Then
            If mediaType.formatSize <> 0 Then
                Marshal.FreeCoTaskMem(mediaType.formatPtr)
                mediaType.formatSize = 0
                mediaType.formatPtr = IntPtr.Zero
            End If
            If mediaType.unkPtr <> IntPtr.Zero Then
                Marshal.Release(mediaType.unkPtr)
                mediaType.unkPtr = IntPtr.Zero
            End If
        End If
    End Sub

    Public Shared Sub FreePinInfo(ByVal pinInfo As PinInfo)
        If pinInfo.filter IsNot Nothing Then
            Marshal.ReleaseComObject(pinInfo.filter)
            pinInfo.filter = Nothing
        End If
    End Sub

End Class


Public Class DsROTEntry
    Implements IDisposable
    <Flags()> _
    Private Enum ROTFlags
        RegistrationKeepsAlive = &H1
        AllowAnyClient = &H2
    End Enum

    Private m_cookie As Integer = 0

    <DllImport("ole32.dll", ExactSpelling:=True), SuppressUnmanagedCodeSecurity()> _
    Private Shared Function GetRunningObjectTable(ByVal r As Integer, ByRef pprot As IRunningObjectTable) As Integer
    End Function

    <DllImport("ole32.dll", CharSet:=CharSet.Unicode, ExactSpelling:=True), SuppressUnmanagedCodeSecurity()> _
    Private Shared Function CreateItemMoniker(ByVal delim As String, ByVal item As String, ByRef ppmk As IMoniker) As Integer
    End Function

    Public Sub New(ByVal graph As IFilterGraph)
        Dim hr As Integer = 0
        Dim rot As IRunningObjectTable = Nothing
        Dim mk As IMoniker = Nothing

        Try
            ' First, get a pointer to the running object table
            hr = GetRunningObjectTable(0, rot)
            DsError.ThrowExceptionForHR(hr)

            ' Build up the object to add to the table
            Dim id As Integer = System.Diagnostics.Process.GetCurrentProcess().Id
            Dim iuPtr As IntPtr = Marshal.GetIUnknownForObject(graph)
            Dim s As String
            Try
                s = iuPtr.ToString("x")
            Catch
                s = ""
            Finally
                Marshal.Release(iuPtr)
            End Try
            Dim item As String = String.Format("FilterGraph {0} pid {1}", s, id.ToString("x8"))
            hr = CreateItemMoniker("!", item, mk)
            DsError.ThrowExceptionForHR(hr)

            ' Add the object to the table

            m_cookie = rot.Register(CInt(ROTFlags.RegistrationKeepsAlive), graph, mk)
        Finally
            If mk IsNot Nothing Then
                Marshal.ReleaseComObject(mk)
                mk = Nothing
            End If
            If rot IsNot Nothing Then
                Marshal.ReleaseComObject(rot)
                rot = Nothing
            End If
        End Try
    End Sub

    Protected Overrides Sub Finalize()
        Try
            Dispose()
        Finally
            MyBase.Finalize()
        End Try
    End Sub

    Public Sub Dispose() Implements IDisposable.Dispose
        If m_cookie <> 0 Then
            GC.SuppressFinalize(Me)
            Dim rot As IRunningObjectTable = Nothing

            ' Get a pointer to the running object table
            Dim hr As Integer = GetRunningObjectTable(0, rot)
            DsError.ThrowExceptionForHR(hr)

            Try
                ' Remove our entry
                rot.Revoke(m_cookie)
                m_cookie = 0
            Finally
                Marshal.ReleaseComObject(rot)
                rot = Nothing
            End Try
        End If
    End Sub
End Class


Public Class DsDevice
    Implements IDisposable
    Private m_Mon As IMoniker
    Private m_Name As String

    Public Sub New(ByVal Mon As IMoniker)
        m_Mon = Mon
        m_Name = Nothing
    End Sub

    Public ReadOnly Property Mon() As IMoniker
        Get
            Return m_Mon
        End Get
    End Property

    Public ReadOnly Property Name() As String
        Get
            If m_Name Is Nothing Then
                m_Name = GetPropBagValue("FriendlyName")
            End If
            Return m_Name
        End Get
    End Property

    Public ReadOnly Property DevicePath() As String
        Get
            Dim s As String = Nothing

            Try
                m_Mon.GetDisplayName(Nothing, Nothing, s)
            Catch
            End Try

            Return s
        End Get
    End Property

    Public ReadOnly Property ClassID() As Guid
        Get
            Dim g As Guid

            m_Mon.GetClassID(g)

            Return g
        End Get
    End Property

    Public Shared Function GetDevicesOfCat(ByVal FilterCategory As Guid) As DsDevice()
        Dim hr As Integer

        ' Use arrayList to build the retun list since it is easily resizable
        Dim devret As DsDevice()
        Dim devs As New ArrayList()
        Dim enumMon As IEnumMoniker = Nothing

        Dim enumDev As ICreateDevEnum = DirectCast(New CreateDevEnum(), ICreateDevEnum)
        hr = enumDev.CreateClassEnumerator(FilterCategory, enumMon, 0)
        DsError.ThrowExceptionForHR(hr)

        ' CreateClassEnumerator returns null for enumMon if there are no entries
        If hr <> 1 Then
            Try
                Try
                    Dim mon As IMoniker() = New IMoniker(0) {}

                    While (enumMon.[Next](1, mon, IntPtr.Zero) = 0)
                        Try
                            ' The devs array now owns this object.  Don't
                            ' release it if we are going to be successfully
                            ' returning the devret array
                            devs.Add(New DsDevice(mon(0)))
                        Catch
                            Marshal.ReleaseComObject(mon(0))
                            Throw
                        End Try
                    End While
                Finally
                    Marshal.ReleaseComObject(enumMon)
                End Try

                ' Copy the ArrayList to the DsDevice[]
                devret = New DsDevice(devs.Count - 1) {}
                devs.CopyTo(devret)
            Catch
                For Each d As DsDevice In devs
                    d.Dispose()
                Next
                Throw
            End Try
        Else
            devret = New DsDevice(-1) {}
        End If

        Return devret
    End Function

    Public Function GetPropBagValue(ByVal sPropName As String) As String
        Dim bag As IPropertyBag = Nothing
        Dim ret As String = Nothing
        Dim bagObj As Object = Nothing
        Dim val As Object = Nothing

        Try
            Dim bagId As Guid = GetType(IPropertyBag).GUID
            m_Mon.BindToStorage(Nothing, Nothing, bagId, bagObj)

            bag = DirectCast(bagObj, IPropertyBag)

            Dim hr As Integer = bag.Read(sPropName, val, Nothing)
            DsError.ThrowExceptionForHR(hr)

            ret = TryCast(val, String)
        Catch
            ret = Nothing
        Finally
            bag = Nothing
            If bagObj IsNot Nothing Then
                Marshal.ReleaseComObject(bagObj)
                bagObj = Nothing
            End If
        End Try

        Return ret
    End Function

    Public Sub Dispose() Implements IDisposable.Dispose
        If Mon IsNot Nothing Then
            Marshal.ReleaseComObject(Mon)
            m_Mon = Nothing
        End If
        m_Name = Nothing
    End Sub
End Class


Public NotInheritable Class DsFindPin
    Private Sub New()
    End Sub

    Public Shared Function ByDirection(ByVal vSource As IBaseFilter, ByVal vDir As PinDirection, ByVal iIndex As Integer) As IPin
        Dim hr As Integer
        Dim ppEnum As IEnumPins = Nothing
        Dim ppindir As PinDirection
        Dim pRet As IPin = Nothing
        Dim pPins As IPin() = New IPin(0) {}

        If vSource Is Nothing Then
            Return Nothing
        End If

        ' Get the pin enumerator
        hr = vSource.EnumPins(ppEnum)
        DsError.ThrowExceptionForHR(hr)

        Try
            ' Walk the pins looking for a match
            While ppEnum.[Next](1, pPins, IntPtr.Zero) = 0
                ' Read the direction
                hr = pPins(0).QueryDirection(ppindir)
                DsError.ThrowExceptionForHR(hr)

                ' Is it the right direction?
                If ppindir = vDir Then
                    ' Is is the right index?
                    If iIndex = 0 Then
                        pRet = pPins(0)
                        Exit While
                    End If
                    iIndex -= 1
                End If
                Marshal.ReleaseComObject(pPins(0))
            End While
        Finally
            Marshal.ReleaseComObject(ppEnum)
        End Try

        Return pRet
    End Function

    Public Shared Function ByName(ByVal vSource As IBaseFilter, ByVal vPinName As String) As IPin
        Dim hr As Integer
        Dim ppEnum As IEnumPins = Nothing
        Dim ppinfo As PinInfo = Nothing
        Dim pRet As IPin = Nothing
        Dim pPins As IPin() = New IPin(0) {}

        If vSource Is Nothing Then
            Return Nothing
        End If

        ' Get the pin enumerator
        hr = vSource.EnumPins(ppEnum)
        DsError.ThrowExceptionForHR(hr)

        Try
            ' Walk the pins looking for a match
            While ppEnum.[Next](1, pPins, IntPtr.Zero) = 0
                ' Read the info
                hr = pPins(0).QueryPinInfo(ppinfo)
                DsError.ThrowExceptionForHR(hr)

                ' Is it the right name?
                If ppinfo.name = vPinName Then
                    DsUtils.FreePinInfo(ppinfo)
                    pRet = pPins(0)
                    Exit While
                End If
                Marshal.ReleaseComObject(pPins(0))
                DsUtils.FreePinInfo(ppinfo)
            End While
        Finally
            Marshal.ReleaseComObject(ppEnum)
        End Try

        Return pRet
    End Function

    Public Shared Function ByCategory(ByVal vSource As IBaseFilter, ByVal PinCategory As Guid, ByVal iIndex As Integer) As IPin
        Dim hr As Integer
        Dim ppEnum As IEnumPins = Nothing
        Dim pRet As IPin = Nothing
        Dim pPins As IPin() = New IPin(0) {}

        If vSource Is Nothing Then
            Return Nothing
        End If

        ' Get the pin enumerator
        hr = vSource.EnumPins(ppEnum)
        DsError.ThrowExceptionForHR(hr)

        Try
            ' Walk the pins looking for a match
            While ppEnum.[Next](1, pPins, IntPtr.Zero) = 0
                ' Is it the right category?
                If DsUtils.GetPinCategory(pPins(0)) = PinCategory Then
                    ' Is is the right index?
                    If iIndex = 0 Then
                        pRet = pPins(0)
                        Exit While
                    End If
                    iIndex -= 1
                End If
                Marshal.ReleaseComObject(pPins(0))
            End While
        Finally
            Marshal.ReleaseComObject(ppEnum)
        End Try

        Return pRet
    End Function

    Public Shared Function ByConnectionStatus(ByVal vSource As IBaseFilter, ByVal vStat As PinConnectedStatus, ByVal iIndex As Integer) As IPin
        Dim hr As Integer
        Dim ppEnum As IEnumPins = Nothing
        Dim pRet As IPin = Nothing
        Dim pOutPin As IPin = Nothing
        Dim pPins As IPin() = New IPin(0) {}

        If vSource Is Nothing Then
            Return Nothing
        End If

        ' Get the pin enumerator
        hr = vSource.EnumPins(ppEnum)
        DsError.ThrowExceptionForHR(hr)

        Try
            ' Walk the pins looking for a match
            While ppEnum.[Next](1, pPins, IntPtr.Zero) = 0
                ' Read the connected status
                hr = pPins(0).ConnectedTo(pOutPin)

                ' Check for VFW_E_NOT_CONNECTED.  Anything else is bad.
                If hr <> DsResults.E_NotConnected Then
                    DsError.ThrowExceptionForHR(hr)

                    ' The ConnectedTo call succeeded, release the interface
                    Marshal.ReleaseComObject(pOutPin)
                End If

                ' Is it the right status?
                If (hr = 0 AndAlso vStat = PinConnectedStatus.Connected) OrElse (hr = DsResults.E_NotConnected AndAlso vStat = PinConnectedStatus.Unconnected) Then
                    ' Is is the right index?
                    If iIndex = 0 Then
                        pRet = pPins(0)
                        Exit While
                    End If
                    iIndex -= 1
                End If
                Marshal.ReleaseComObject(pPins(0))
            End While
        Finally
            Marshal.ReleaseComObject(ppEnum)
        End Try

        Return pRet
    End Function
End Class


Public NotInheritable Class DsToString
    Private Sub New()
    End Sub

    Public Shared Function AMMediaTypeToString(ByVal pmt As AMMediaType) As String
        Return String.Format("{0} {1} {2} {3} {4} {5}", MediaTypeToString(pmt.majorType), MediaSubTypeToString(pmt.subType), MediaFormatTypeToString(pmt.formatType), (If(pmt.fixedSizeSamples, "FixedSamples", "NotFixedSamples")), (If(pmt.temporalCompression, "temporalCompression", "NottemporalCompression")), _
         pmt.sampleSize.ToString())
    End Function

   Public Shared Function MediaTypeToString(ByVal guid As Guid) As String
        ' Walk the MediaSubType class looking for a match
        Return WalkClass(GetType(MediaType), guid)
    End Function

    Public Shared Function MediaSubTypeToString(ByVal guid As Guid) As String
        ' Walk the MediaSubType class looking for a match
        Dim s As String = WalkClass(GetType(MediaSubType), guid)

        ' There is a special set of Guids that contain the FourCC code
        ' as part of the Guid.  Check to see if it is one of those.
        If s.Length = 36 AndAlso s.Substring(8).ToUpper() = "-0000-0010-8000-00AA00389B71" Then
            ' Parse out the FourCC code
            Dim asc As Byte() = {Convert.ToByte(s.Substring(6, 2), 16), Convert.ToByte(s.Substring(4, 2), 16), Convert.ToByte(s.Substring(2, 2), 16), Convert.ToByte(s.Substring(0, 2), 16)}
            s = Encoding.ASCII.GetString(asc)
        End If

        Return s
    End Function

    Public Shared Function MediaFormatTypeToString(ByVal guid As Guid) As String
        ' Walk the FormatType class looking for a match
        Return WalkClass(GetType(FormatType), guid)

    End Function

    Private Shared Function WalkClass(ByVal MyType As Type, ByVal guid As Guid) As String
        Dim o As Object = Nothing

        ' Read the fields from the class
        Dim Fields As FieldInfo() = MyType.GetFields()

        ' Walk the returned array
        For Each m As FieldInfo In Fields
            ' Read the value of the property.  The parameter is ignored.
            o = m.GetValue(o)

            ' Compare it with the sought value
            If CType(o, Guid) = guid Then
                Return m.Name
            End If
        Next

        Return guid.ToString()
    End Function
End Class


Friend MustInherit Class DsMarshaler
    Implements ICustomMarshaler

    ' The cookie isn't currently being used.
    Protected m_cookie As String

    ' The managed object passed in to MarshalManagedToNative, and modified in MarshalNativeToManaged
    Protected m_obj As Object

    ' The constructor.  This is called from GetInstance (below)
    Public Sub New(ByVal cookie As String)
        ' If we get a cookie, save it.
        m_cookie = cookie
    End Sub

    ' Called just before invoking the COM method.  The returned IntPtr is what goes on the stack
    ' for the COM call.  The input arg is the parameter that was passed to the method.
    Public Overridable Function MarshalManagedToNative(ByVal managedObj As Object) As IntPtr Implements ICustomMarshaler.MarshalManagedToNative
        ' Save off the passed-in value.  Safe since we just checked the type.
        m_obj = managedObj

        ' Create an appropriately sized buffer, blank it, and send it to the marshaler to
        ' make the COM call with.
        Dim iSize As Integer = GetNativeDataSize() + 3
        Dim p As IntPtr = Marshal.AllocCoTaskMem(iSize)

        For x As Integer = 0 To iSize \ 4 - 1
            Marshal.WriteInt32(p, x * 4, 0)
        Next

        Return p
    End Function

    ' Called just after invoking the COM method.  The IntPtr is the same one that just got returned
    ' from MarshalManagedToNative.  The return value is unused.
    Public Overridable Function MarshalNativeToManaged(ByVal pNativeData As IntPtr) As Object Implements ICustomMarshaler.MarshalNativeToManaged
        Return m_obj
    End Function

    ' Release the (now unused) buffer
    Public Overridable Sub CleanUpNativeData(ByVal pNativeData As IntPtr) Implements ICustomMarshaler.CleanUpNativeData
        If pNativeData <> IntPtr.Zero Then
            Marshal.FreeCoTaskMem(pNativeData)
        End If
    End Sub

    ' Release the (now unused) managed object
    Public Overridable Sub CleanUpManagedData(ByVal managedObj As Object) Implements ICustomMarshaler.CleanUpManagedData
        m_obj = Nothing
    End Sub

    ' This routine is (apparently) never called by the marshaler.  However it can be useful.
    Public MustOverride Function GetNativeDataSize() As Integer

    ' GetInstance is called by the marshaler in preparation to doing custom marshaling.  The (optional)
    ' cookie is the value specified in MarshalCookie="asdf", or "" is none is specified.

    ' It is commented out in this abstract class, but MUST be implemented in derived classes
    'public static ICustomMarshaler GetInstance(string cookie)

    Public Function GetNativeDataSize1() As Integer Implements System.Runtime.InteropServices.ICustomMarshaler.GetNativeDataSize

    End Function
End Class


Friend Class EMTMarshaler
    Inherits DsMarshaler
    Public Sub New(ByVal cookie As String)
        MyBase.New(cookie)
    End Sub

    ' Called just after invoking the COM method.  The IntPtr is the same one that just got returned
    ' from MarshalManagedToNative.  The return value is unused.
    Public Overrides Function MarshalNativeToManaged(ByVal pNativeData As IntPtr) As Object
        Dim emt As AMMediaType() = TryCast(m_obj, AMMediaType())

        For x As Integer = 0 To emt.Length - 1
            ' Copy in the value, and advance the pointer
            Dim p As IntPtr = Marshal.ReadIntPtr(pNativeData, x * IntPtr.Size)
            If p <> IntPtr.Zero Then
                emt(x) = DirectCast(Marshal.PtrToStructure(p, GetType(AMMediaType)), AMMediaType)
            Else
                emt(x) = Nothing
            End If
        Next

        Return Nothing
    End Function

    ' The number of bytes to marshal out
    Public Overrides Function GetNativeDataSize() As Integer
        ' Get the array size
        Dim i As Integer = DirectCast(m_obj, Array).Length

        ' Multiply that times the size of a pointer
        Dim j As Integer = i * IntPtr.Size

        Return j
    End Function

    ' This method is called by interop to create the custom marshaler.  The (optional)
    ' cookie is the value specified in MarshalCookie="asdf", or "" is none is specified.
    Public Shared Function GetInstance(ByVal cookie As String) As ICustomMarshaler
        Return New EMTMarshaler(cookie)
    End Function
End Class

' c# does not correctly create structures that contain ByValArrays of structures (or enums!).  Instead
' of allocating enough room for the ByValArray of structures, it only reserves room for a ref,
' even when decorated with ByValArray and SizeConst.  Needless to say, if DirectShow tries to
' write to this too-short buffer, bad things will happen.
'
' To work around this for the DvdTitleAttributes structure, use this custom marshaler
' by declaring the parameter DvdTitleAttributes as:
'
'    [In, Out, MarshalAs(UnmanagedType.CustomMarshaler, MarshalTypeRef=typeof(DTAMarshaler))]
'    DvdTitleAttributes pTitle
'
' See DsMarshaler for more info on custom marshalers

Friend Class DTAMarshaler
    Inherits DsMarshaler
    Public Sub New(ByVal cookie As String)
        MyBase.New(cookie)
    End Sub

    ' Called just after invoking the COM method.  The IntPtr is the same one that just got returned
    ' from MarshalManagedToNative.  The return value is unused.
    Public Overrides Function MarshalNativeToManaged(ByVal pNativeData As IntPtr) As Object
        Dim dta As DvdTitleAttributes = TryCast(m_obj, DvdTitleAttributes)

        ' Copy in the value, and advance the pointer
        dta.AppMode = CType(Marshal.ReadInt32(pNativeData), DvdTitleAppMode)
        pNativeData = CType(pNativeData.ToInt64() + Marshal.SizeOf(GetType(Integer)), IntPtr)

        ' Copy in the value, and advance the pointer
        dta.VideoAttributes = CType(Marshal.PtrToStructure(pNativeData, GetType(DvdVideoAttributes)), DvdVideoAttributes)
        pNativeData = CType(pNativeData.ToInt64() + Marshal.SizeOf(GetType(DvdVideoAttributes)), IntPtr)

        ' Copy in the value, and advance the pointer
        dta.ulNumberOfAudioStreams = Marshal.ReadInt32(pNativeData)
        pNativeData = CType(pNativeData.ToInt64() + Marshal.SizeOf(GetType(Integer)), IntPtr)

        ' Allocate a large enough array to hold all the returned structs.
        dta.AudioAttributes = New DvdAudioAttributes(7) {}
        For x As Integer = 0 To 7
            ' Copy in the value, and advance the pointer
            dta.AudioAttributes(x) = CType(Marshal.PtrToStructure(pNativeData, GetType(DvdAudioAttributes)), DvdAudioAttributes)
            pNativeData = CType(pNativeData.ToInt64() + Marshal.SizeOf(GetType(DvdAudioAttributes)), IntPtr)
        Next

        ' Allocate a large enough array to hold all the returned structs.
        dta.MultichannelAudioAttributes = New DvdMultichannelAudioAttributes(7) {}
        For x As Integer = 0 To 7
            ' MultichannelAudioAttributes has nested ByValArrays.  They need to be individually copied.

            dta.MultichannelAudioAttributes(x).Info = New DvdMUAMixingInfo(7) {}

            For y As Integer = 0 To 7
                ' Copy in the value, and advance the pointer
                dta.MultichannelAudioAttributes(x).Info(y) = CType(Marshal.PtrToStructure(pNativeData, GetType(DvdMUAMixingInfo)), DvdMUAMixingInfo)
                pNativeData = CType(pNativeData.ToInt64() + Marshal.SizeOf(GetType(DvdMUAMixingInfo)), IntPtr)
            Next

            dta.MultichannelAudioAttributes(x).Coeff = New DvdMUACoeff(7) {}

            For y As Integer = 0 To 7
                ' Copy in the value, and advance the pointer
                dta.MultichannelAudioAttributes(x).Coeff(y) = CType(Marshal.PtrToStructure(pNativeData, GetType(DvdMUACoeff)), DvdMUACoeff)
                pNativeData = CType(pNativeData.ToInt64() + Marshal.SizeOf(GetType(DvdMUACoeff)), IntPtr)
            Next
        Next

        ' The DvdMultichannelAudioAttributes needs to be 16 byte aligned
        pNativeData = CType(pNativeData.ToInt64() + 4, IntPtr)

        ' Copy in the value, and advance the pointer
        dta.ulNumberOfSubpictureStreams = Marshal.ReadInt32(pNativeData)
        pNativeData = CType(pNativeData.ToInt64() + Marshal.SizeOf(GetType(Integer)), IntPtr)

        ' Allocate a large enough array to hold all the returned structs.
        dta.SubpictureAttributes = New DvdSubpictureAttributes(31) {}
        For x As Integer = 0 To 31
            ' Copy in the value, and advance the pointer
            dta.SubpictureAttributes(x) = CType(Marshal.PtrToStructure(pNativeData, GetType(DvdSubpictureAttributes)), DvdSubpictureAttributes)
            pNativeData = CType(pNativeData.ToInt64() + Marshal.SizeOf(GetType(DvdSubpictureAttributes)), IntPtr)
        Next

        ' Note that 4 bytes (more alignment) are unused at the end

        Return Nothing
    End Function

    ' The number of bytes to marshal out
    Public Overrides Function GetNativeDataSize() As Integer
        ' This is the actual size of a DvdTitleAttributes structure
        Return 3208
    End Function

    ' This method is called by interop to create the custom marshaler.  The (optional)
    ' cookie is the value specified in MarshalCookie="asdf", or "" is none is specified.
    Public Shared Function GetInstance(ByVal cookie As String) As ICustomMarshaler
        Return New DTAMarshaler(cookie)
    End Function
End Class

' See DTAMarshaler for an explanation of the problem.  This class is for marshaling
' a DvdKaraokeAttributes structure.
Friend Class DKAMarshaler
    Inherits DsMarshaler
    ' The constructor.  This is called from GetInstance (below)
    Public Sub New(ByVal cookie As String)
        MyBase.New(cookie)
    End Sub

    ' Called just after invoking the COM method.  The IntPtr is the same one that just got returned
    ' from MarshalManagedToNative.  The return value is unused.
    Public Overrides Function MarshalNativeToManaged(ByVal pNativeData As IntPtr) As Object
        Dim dka As DvdKaraokeAttributes = TryCast(m_obj, DvdKaraokeAttributes)

        ' Copy in the value, and advance the pointer
        dka.bVersion = CByte(Marshal.ReadByte(pNativeData))
        pNativeData = CType(pNativeData.ToInt64() + Marshal.SizeOf(GetType(Byte)), IntPtr)

        ' DWORD Align
        pNativeData = CType(pNativeData.ToInt64() + 3, IntPtr)

        ' Copy in the value, and advance the pointer
        dka.fMasterOfCeremoniesInGuideVocal1 = Marshal.ReadInt32(pNativeData) <> 0
        pNativeData = CType(pNativeData.ToInt64() + Marshal.SizeOf(GetType(Boolean)), IntPtr)

        ' Copy in the value, and advance the pointer
        dka.fDuet = Marshal.ReadInt32(pNativeData) <> 0
        pNativeData = CType(pNativeData.ToInt64() + Marshal.SizeOf(GetType(Boolean)), IntPtr)

        ' Copy in the value, and advance the pointer
        dka.ChannelAssignment = CType(Marshal.ReadInt32(pNativeData), DvdKaraokeAssignment)
        pNativeData = CType(pNativeData.ToInt64() + Marshal.SizeOf(DvdKaraokeAssignment.GetUnderlyingType(GetType(DvdKaraokeAssignment))), IntPtr)

        ' Allocate a large enough array to hold all the returned structs.
        dka.wChannelContents = New DvdKaraokeContents(7) {}
        For x As Integer = 0 To 7
            ' Copy in the value, and advance the pointer
            dka.wChannelContents(x) = CType(Marshal.ReadInt16(pNativeData), DvdKaraokeContents)
            pNativeData = CType(pNativeData.ToInt64() + Marshal.SizeOf(DvdKaraokeContents.GetUnderlyingType(GetType(DvdKaraokeContents))), IntPtr)
        Next

        Return Nothing
    End Function

    ' The number of bytes to marshal out
    Public Overrides Function GetNativeDataSize() As Integer
        ' This is the actual size of a DvdKaraokeAttributes structure.
        Return 32
    End Function

    ' This method is called by interop to create the custom marshaler.  The (optional)
    ' cookie is the value specified in MarshalCookie="asdf", or "" is none is specified.
    Public Shared Function GetInstance(ByVal cookie As String) As ICustomMarshaler
        Return New DKAMarshaler(cookie)
    End Function

End Class

