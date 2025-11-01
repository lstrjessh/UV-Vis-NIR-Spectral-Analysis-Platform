
Imports System
Imports System.Diagnostics
Imports System.Drawing
Imports System.Runtime.InteropServices


Friend Class VideoCapabilities

    Private VideoCaps() As AnalogVideoCaps_Struct

    Private Structure AnalogVideoCaps_Struct
        ' ------------------------------------------------- I420 / RGB24 etc...
        Friend MediaSubType As String

        ' ------------------------------------------------- Native size of the incoming video signal
        '                                                   ( largest signal the filter can digitize 
        '                                                           with every pixel remaining unique )
        Friend StringInputSize As String

        ' ------------------------------------------------- Min / Max supported frame rate
        Friend MinFrameRate As Double
        Friend MaxFrameRate As Double

        ' ------------------------------------------------- Analog video standard(s) 
        Friend AnalogVideoStandard As AnalogVideoStandard
    End Structure

    Friend Sub New(ByVal videoStreamConfig As IAMStreamConfig)
        '
        If (videoStreamConfig Is Nothing) Then Exit Sub
        '
        Dim mediaType As AMMediaType = Nothing
        Dim caps As VideoStreamConfigCaps = Nothing
        Dim pCaps As IntPtr = IntPtr.Zero
        '
        Try
            Dim size As Integer
            Dim count As Integer
            videoStreamConfig.GetNumberOfCapabilities(count, size)
            '
            ReDim VideoCaps(count - 1)
            '
            If count > 0 And size = Marshal.SizeOf(GetType(VideoStreamConfigCaps)) Then

                ' --------------------------------------------------------- Alloc memory for structure
                pCaps = Marshal.AllocCoTaskMem(Marshal.SizeOf(GetType(VideoStreamConfigCaps)))

                ' --------------------------------------------------------- read all structures
                For index As Int32 = 0 To count - 1
                    videoStreamConfig.GetStreamCaps(index, mediaType, pCaps)

                    ' ----------------------------------------------------- mediaType.subType GUID to string
                    VideoCaps(index).MediaSubType = MediaSubTypeToString(mediaType.subType)

                    ' ----------------------------------------------------- Convert pointers to managed structures
                    caps = CType(Marshal.PtrToStructure(pCaps, _
                                                        GetType(VideoStreamConfigCaps)),  _
                                                        VideoStreamConfigCaps)

                    ' Extract info
                    VideoCaps(index).StringInputSize = caps.InputSize.Width.ToString & " x " & caps.InputSize.Height.ToString
                    '
                    VideoCaps(index).MinFrameRate = (CType(10000000, Double) / caps.MaxFrameInterval)
                    VideoCaps(index).MaxFrameRate = (CType(10000000, Double) / caps.MinFrameInterval)
                    '
                    VideoCaps(index).AnalogVideoStandard = caps.VideoStandard

                    ' deprecated
                    'VideoCaps(index).MinFrameSize = caps.MinOutputSize
                    'VideoCaps(index).MaxFrameSize = caps.MaxOutputSize
                    'VideoCaps(index).FrameSizeGranularityX = caps.OutputGranularityX
                    'VideoCaps(index).FrameSizeGranularityY = caps.OutputGranularityY
                    '
                Next
            End If

        Finally
            If pCaps <> IntPtr.Zero Then Marshal.FreeCoTaskMem(pCaps) : pCaps = IntPtr.Zero
            If mediaType IsNot Nothing Then DsUtils.FreeAMMediaType(mediaType) : mediaType = Nothing
        End Try
    End Sub

    Friend Function ValidFormats(ByVal VideoSize As String) As String()
        If VideoCaps Is Nothing Then Return New String() {""}
        Dim strArray(-1) As String
        For i As Int32 = 0 To VideoCaps.Count - 1
            If VideoSize = VideoCaps(i).StringInputSize And Not strArray.Contains(VideoCaps(i).MediaSubType) Then
                ReDim Preserve strArray(strArray.Count)
                strArray(strArray.Count - 1) = VideoCaps(i).MediaSubType
            End If
        Next
        Array.Sort(strArray)
        Return strArray
    End Function

    Friend Function ValidSizes(ByVal MediaSubType As String) As String()
        If VideoCaps Is Nothing Then Return New String() {""}
        Dim strArray(-1) As String
        For i As Int32 = 0 To VideoCaps.Count - 1
            If MediaSubType = VideoCaps(i).MediaSubType Then
                ReDim Preserve strArray(strArray.Count)
                strArray(strArray.Count - 1) = VideoCaps(i).StringInputSize
            End If
        Next
        Array.Sort(strArray, AddressOf StringNumericComparer)
        Return strArray
    End Function
    Private Function StringNumericComparer(ByVal x As String, ByVal y As String) As Integer
        Return Math.Sign(Val(x) - Val(y))
    End Function

    Friend Function ValidFPS(ByVal MediaSubType As String, ByVal VideoSize As String) As String()
        If VideoCaps Is Nothing Then Return New String() {""}
        Dim strArray(-1) As String
        For i As Int32 = 0 To VideoCaps.Count - 1
            If MediaSubType = VideoCaps(i).MediaSubType And VideoSize = VideoCaps(i).StringInputSize Then
                '
                Dim minfps As Int32 = CInt(VideoCaps(i).MinFrameRate)
                Dim maxfps As Int32 = CInt(VideoCaps(i).MaxFrameRate)
                If minfps < maxfps And minfps > 3 Then minfps = 3
                '
                For fps As Int32 = minfps To maxfps Step 3
                    ReDim Preserve strArray(strArray.Count)
                    strArray(strArray.Count - 1) = fps.ToString
                Next
                '
                Exit For
            End If
        Next
        Return strArray
    End Function

End Class










' ========================================================================================================= 
' ORIGINAL VB
' =========================================================================================================

'Friend Class VideoCapabilities

'    ' Analog video standard(s) 
'    Public AnalogVideoStandard As AnalogVideoStandard

'    ' Native size of the incoming video signal. This is the largest signal the filter can digitize with every pixel remaining unique. Read-only.
'    Public InputSize As Size

'    ' Minimum supported frame size. Read-only.
'    Public MinFrameSize As Size

'    ' Maximum supported frame size. Read-only.
'    Public MaxFrameSize As Size

'    ' Granularity of the output width. This value specifies the increments that are valid between MinFrameSize and MaxFrameSize. Read-only. 
'    Public FrameSizeGranularityX As Integer

'    ' Granularity of the output height. This value specifies the increments that are valid between MinFrameSize and MaxFrameSize. Read-only. 
'    Public FrameSizeGranularityY As Integer

'    ' Minimum supported frame rate. Read-only. 
'    Public MinFrameRate As Double

'    ' Maximum supported frame rate. Read-only. 
'    Public MaxFrameRate As Double

'    ' ----------------- Constructor ---------------------
'    'Retrieve capabilities of a video device
'    Friend Sub New(ByVal videoStreamConfig As IAMStreamConfig)
'        MyBase.New()
'        If (videoStreamConfig Is Nothing) Then
'            Throw New ArgumentNullException("videoStreamConfig")
'        End If
'        Dim mediaType As AMMediaType = Nothing
'        Dim caps As VideoStreamConfigCaps = Nothing
'        Dim pCaps As IntPtr = IntPtr.Zero

'        'Dim pMediaType As IntPtr

'        Try
'            ' Ensure this device reports capabilities
'            Dim size As Integer
'            Dim c As Integer
'            Dim hr As Integer = videoStreamConfig.GetNumberOfCapabilities(c, size)
'            If (hr <> 0) Then
'                Marshal.ThrowExceptionForHR(hr)
'            End If
'            If (c <= 0) Then
'                Throw New NotSupportedException("This video device does not report capabilities.")
'            End If
'            If (size > Marshal.SizeOf(GetType(VideoStreamConfigCaps))) Then
'                Throw New NotSupportedException("Unable to retrieve video device capabilities. This video device requires a larger VideoStreamConfigCa" & _
'                    "ps structure.")
'            End If
'            If (c > 1) Then
'                Debug.WriteLine(("This video device supports " _
'                                & (c & " capability structures. Only the first structure will be used.")))
'            End If
'            ' Alloc memory for structure
'            pCaps = Marshal.AllocCoTaskMem(Marshal.SizeOf(GetType(VideoStreamConfigCaps)))
'            ' Retrieve first (and hopefully only) capabilities struct

'            ' hr = videoStreamConfig.GetStreamCaps(0, pMediaType, pCaps)

'            hr = videoStreamConfig.GetStreamCaps(0, mediaType, pCaps)

'            If (hr <> 0) Then
'                Marshal.ThrowExceptionForHR(hr)
'            End If

'            'mediaType = CType(Marshal.PtrToStructure(pMediaType, GetType(AMMediaType)),AMMediaType)

'            ' Convert pointers to managed structures
'            caps = CType(Marshal.PtrToStructure(pCaps, GetType(VideoStreamConfigCaps)), VideoStreamConfigCaps)
'            ' Extract info
'            InputSize = caps.InputSize
'            MinFrameSize = caps.MinOutputSize
'            MaxFrameSize = caps.MaxOutputSize
'            FrameSizeGranularityX = caps.OutputGranularityX
'            FrameSizeGranularityY = caps.OutputGranularityY
'            MinFrameRate = (CType(10000000, Double) / caps.MaxFrameInterval)
'            MaxFrameRate = (CType(10000000, Double) / caps.MinFrameInterval)
'            '#if NEWCODE
'            Me.AnalogVideoStandard = caps.VideoStandard
'            If (caps.VideoStandard > AnalogVideoStandard.None) Then
'                Debug.WriteLine((caps.InputSize.ToString + (" " _
'                                + (caps.MinOutputSize.ToString + (" " _
'                                + (caps.MaxOutputSize.ToString + (" " _
'                                + (MinFrameRate.ToString + ("-" _
'                                + (MaxFrameRate.ToString + (" " _
'                                + (caps.VideoStandard & (" - " & mediaType.ToString)))))))))))))
'            End If
'            '#endif
'        Finally
'            If (pCaps <> IntPtr.Zero) Then
'                Marshal.FreeCoTaskMem(pCaps)
'            End If
'            pCaps = IntPtr.Zero
'            If (Not (mediaType) Is Nothing) Then
'                DsUtils.FreeAMMediaType(mediaType)
'            End If
'            mediaType = Nothing
'        End Try
'    End Sub
'End Class





' ========================================================================================================= 
' ORIGINAL C#
' =========================================================================================================

'using System;
'using System.Diagnostics;
'using System.Drawing;
'using System.Runtime.InteropServices;
'using DShowNET;

'Namespace DirectX.Capture
'{
'	/// <summary>
'	///  Capabilities of the video device such as 
'	///  min/max frame size and frame rate.
'	/// </summary>
'    Public Class VideoCapabilities
'	{

'		// ------------------ Properties --------------------

'		/// <summary> Native size of the incoming video signal. This is the largest signal the filter can digitize with every pixel remaining unique. Read-only. </summary>
'		public Size InputSize;

'		/// <summary> Minimum supported frame size. Read-only. </summary>
'		public Size MinFrameSize;

'		/// <summary> Maximum supported frame size. Read-only. </summary>
'		public Size MaxFrameSize;

'		/// <summary> Granularity of the output width. This value specifies the increments that are valid between MinFrameSize and MaxFrameSize. Read-only. </summary>
'		public int FrameSizeGranularityX;

'		/// <summary> Granularity of the output height. This value specifies the increments that are valid between MinFrameSize and MaxFrameSize. Read-only. </summary>
'		public int FrameSizeGranularityY;

'		/// <summary> Minimum supported frame rate. Read-only. </summary>
'		public double MinFrameRate;

'		/// <summary> Maximum supported frame rate. Read-only. </summary>
'		public double MaxFrameRate;



'		// ----------------- Constructor ---------------------

'		/// <summary> Retrieve capabilities of a video device </summary>
'		internal VideoCapabilities(IAMStreamConfig videoStreamConfig)
'		{
'			if ( videoStreamConfig == null ) 
'				throw new ArgumentNullException( "videoStreamConfig" );

'			AMMediaType mediaType = null;
'			VideoStreamConfigCaps caps = null;
'			IntPtr pCaps = IntPtr.Zero;
'			IntPtr pMediaType;
'			try
'			{
'				// Ensure this device reports capabilities
'				int c, size;
'				int hr = videoStreamConfig.GetNumberOfCapabilities( out c, out size );
'				if ( hr != 0 ) Marshal.ThrowExceptionForHR( hr );
'				if ( c <= 0 ) 
'					throw new NotSupportedException( "This video device does not report capabilities." );
'				if ( size > Marshal.SizeOf( typeof( VideoStreamConfigCaps ) ) )
'					throw new NotSupportedException( "Unable to retrieve video device capabilities. This video device requires a larger VideoStreamConfigCaps structure." );
'				if ( c > 1 )
'					Debug.WriteLine("This video device supports " + c + " capability structures. Only the first structure will be used." );

'				// Alloc memory for structure
'				pCaps = Marshal.AllocCoTaskMem( Marshal.SizeOf( typeof( VideoStreamConfigCaps ) ) ); 

'				// Retrieve first (and hopefully only) capabilities struct
'				hr = videoStreamConfig.GetStreamCaps( 0, out pMediaType, pCaps );
'				if ( hr != 0 ) Marshal.ThrowExceptionForHR( hr );

'				// Convert pointers to managed structures
'				mediaType = (AMMediaType) Marshal.PtrToStructure( pMediaType, typeof( AMMediaType ) );
'				caps = (VideoStreamConfigCaps) Marshal.PtrToStructure( pCaps, typeof( VideoStreamConfigCaps ) );

'				// Extract info
'				InputSize = caps.InputSize;
'				MinFrameSize = caps.MinOutputSize;
'				MaxFrameSize = caps.MaxOutputSize;
'				FrameSizeGranularityX = caps.OutputGranularityX;
'				FrameSizeGranularityY = caps.OutputGranularityY;
'				MinFrameRate = (double)10000000 / caps.MaxFrameInterval;
'				MaxFrameRate = (double)10000000 / caps.MinFrameInterval;
'			}
'			finally
'			{
'				if ( pCaps != IntPtr.Zero )
'					Marshal.FreeCoTaskMem( pCaps ); pCaps = IntPtr.Zero;
'				if ( mediaType != null )
'					DsUtils.FreeAMMediaType( mediaType ); mediaType = null;
'			}
'		}
'	}
'}
