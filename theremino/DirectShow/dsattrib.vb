
Imports System.Runtime.InteropServices

Namespace BDA

#If ALLOW_UNTESTED_INTERFACES Then

	Public Enum SAMPLE_SEQ
		SequenceHeader = 1
		GOPHeader = 2
		PictureHeader = 3

		SequenceStart = 1
		SeekPoint = 2
		FrameStart = 3
	End Enum

	Public Enum SAMPLE_SEQ_CONTENT
		Unknown = 0
		IFrame = 1
		PFrame = 2
		BFrame = 3
		StandAloneFrame = 1
		RefFrame = 2
		NonRefFrame = 3
	End Enum

	Public Enum VA_VIDEO_FORMAT
		Component = 0
		PAL = 1
		NTSC = 2
		SECAM = 3
		MAC = 4

		Unspecified = 5
	End Enum

	Public Enum VA_COLOR_PRIMARIES
		ITU_R_BT_709 = 1
		UNSPECIFIED = 2
		ITU_R_BT_470_SYSTEM_M = 4
		ITU_R_BT_470_SYSTEM_B_G = 5
		SMPTE_170M = 6
		SMPTE_240M = 7
		H264_GENERIC_FILM = 8
	End Enum

	Public Enum VA_TRANSFER_CHARACTERISTICS
		ITU_R_BT_709 = 1
		UNSPECIFIED = 2
		ITU_R_BT_470_SYSTEM_M = 4
		ITU_R_BT_470_SYSTEM_B_G = 5
		SMPTE_170M = 6
		SMPTE_240M = 7
		LINEAR = 8
		H264_LOG_100_TO_1 = 9
		H264_LOG_316_TO_1 = 10
	End Enum

	Public Enum VA_MATRIX_COEFFICIENTS
		H264_RGB = 0
		ITU_R_BT_709 = 1
		UNSPECIFIED = 2
		FCC = 4
		ITU_R_BT_470_SYSTEM_B_G = 5
		SMPTE_170M = 6
		SMPTE_240M = 7
		H264_YCgCo = 8
	End Enum

	<StructLayout(LayoutKind.Sequential)> _
	Public Class UDCR_TAG
		Public bVersion As Byte
		<MarshalAs(UnmanagedType.ByValArray, SizeConst := 25)> _
		Public KID As Byte
		Public ullBaseCounter As Long
		Public ullBaseCounterRange As Long
		Public fScrambled As Boolean
		Public bStreamMark As Byte
		Public dwReserved1 As Integer
		Public dwReserved2 As Integer
	End Class

	<StructLayout(LayoutKind.Sequential)> _
	Public Class VA_OPTIONAL_VIDEO_PROPERTIES
		Public dwPictureHeight As Short
		Public dwPictureWidth As Short
		Public dwAspectRatioX As Short
		Public dwAspectRatioY As Short
		Public VAVideoFormat As VA_VIDEO_FORMAT
		Public VAColorPrimaries As VA_COLOR_PRIMARIES
		Public VATransferCharacteristics As VA_TRANSFER_CHARACTERISTICS
		Public VAMatrixCoefficients As VA_MATRIX_COEFFICIENTS
	End Class

	<StructLayout(LayoutKind.Sequential)> _
	Public Class TRANSPORT_PROPERTIES
		Public PID As Integer
		Public PCR As Long
		Public Value As Long
	End Class

	<StructLayout(LayoutKind.Sequential)> _
	Public Class PBDA_TAG_ATTRIBUTE
		Public TableUUId As Guid
		Public TableId As Byte
		Public VersionNo As Short
		Public TableDataSize As Integer
		Public TableData As Byte        ' Array of bytes
	End Class

	<StructLayout(LayoutKind.Sequential)> _
	Public Class CAPTURE_STREAMTIME
		Public StreamTime As Long
	End Class

	<StructLayout(LayoutKind.Sequential)> _
	Public Class DSHOW_STREAM_DESC
		Public VersionNo As Integer
		Public StreamId As Integer
		Public [Default] As Boolean
		Public Creation As Boolean
		Public Reserved As Integer
	End Class

	<StructLayout(LayoutKind.Sequential)> _
	Public Class SAMPLE_LIVE_STREAM_TIME
		Public qwStreamTime As Long
		Public qwLiveTime As Long
	End Class

#End If


#If ALLOW_UNTESTED_INTERFACES Then

	<ComImport, System.Security.SuppressUnmanagedCodeSecurity, Guid("583ec3cc-4960-4857-982b-41a33ea0a006"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)> _
	Public Interface IAttributeSet
		<PreserveSig> _
		Function SetAttrib(<[In]> guidAttribute As Guid, <[In]> pbAttribute As IntPtr, <[In]> dwAttributeLength As Integer) As Integer
	End Interface

	<ComImport, System.Security.SuppressUnmanagedCodeSecurity, Guid("52dbd1ec-e48f-4528-9232-f442a68f0ae1"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)> _
	Public Interface IAttributeGet
		<PreserveSig> _
		Function GetCount(<Out> ByRef plCount As Integer) As Integer

		<PreserveSig> _
		Function GetAttribIndexed(<[In]> lIndex As Integer, <Out> ByRef guidAttribute As Guid, <[In], Out> pbAttribute As IntPtr, <[In], Out> ByRef dwAttributeLength As Integer) As Integer

		<PreserveSig> _
		Function GetAttrib(<[In]> guidAttribute As Guid, <[In], Out> pbAttribute As IntPtr, <[In], Out> ByRef dwAttributeLength As Integer) As Integer
	End Interface

#End If

End Namespace
