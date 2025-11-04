
Imports System.Runtime.InteropServices

Namespace BDA

    <StructLayout(LayoutKind.Sequential, Pack:=2)> _
    Public Structure PidBits
        Public Bits As Short

        Public ReadOnly Property Reserved() As Short
            Get
                Return CShort(CInt(Bits) And &H7)
            End Get
        End Property

        Public ReadOnly Property ProgramId() As Short
            Get
                Return CShort((CInt(Bits) And &HFFF8) >> 3)
            End Get
        End Property
    End Structure

    <StructLayout(LayoutKind.Sequential, Pack:=2)> _
    Public Structure MpegHeaderBits
        Public Bits As Short

        Public ReadOnly Property SectionLength() As Short
            Get
                Return CShort(CInt(Bits) And &HFFF)
            End Get
        End Property

        Public ReadOnly Property Reserved() As Short
            Get
                Return CShort((CInt(Bits) And &H3000) >> 12)
            End Get
        End Property

        Public ReadOnly Property PrivateIndicator() As Short
            Get
                Return CShort((CInt(Bits) And &H4000) >> 14)
            End Get
        End Property

        Public ReadOnly Property SectionSyntaxIndicator() As Short
            Get
                Return CShort((CInt(Bits) And &H8000) >> 15)
            End Get
        End Property
    End Structure

    <StructLayout(LayoutKind.Sequential, Pack:=1)> _
    Public Structure MpegHeaderVersionBits
        Public Bits As Byte

        Public ReadOnly Property CurrentNextIndicator() As Byte
            Get
                Return CByte(CInt(Bits) And &H1)
            End Get
        End Property

        Public ReadOnly Property VersionNumber() As Byte
            Get
                Return CByte((CInt(Bits) And &H3E) >> 1)
            End Get
        End Property

        Public ReadOnly Property Reserved() As Byte
            Get
                Return CByte((CInt(Bits) And &HC0) >> 6)
            End Get
        End Property
    End Structure

#If ALLOW_UNTESTED_INTERFACES Then

	Public Enum MpegSectionIs
		[Next] = 0
		Current = 1
	End Enum

	<StructLayout(LayoutKind.Sequential, Pack := 2)> _
	Public Structure TidExtension
		Public wTidExt As Short
		Public wCount As Short
	End Structure

	<StructLayout(LayoutKind.Sequential, Pack := 1)> _
	Public Structure Section
		Public TableId As Short
		Public Header As MpegHeaderBits
		Public SectionData As Byte
		' Must be marshalled manually
	End Structure

	<StructLayout(LayoutKind.Sequential, Pack := 1)> _
	Public Structure LongSection
		Public TableId As Short
		Public Header As MpegHeaderBits
		Public TableIdExtension As Short
		Public Version As MpegHeaderVersionBits
		Public SectionNumber As Byte
		Public LastSectionNumber As Byte
		Public RemainingData As Byte
		' Must be marshalled manually
	End Structure

	<StructLayout(LayoutKind.Sequential, Pack := 1)> _
	Public Structure DsmccSection
		Public TableId As Short
		Public Header As MpegHeaderBits
		Public TableIdExtension As Short
		Public Version As MpegHeaderVersionBits
		Public SectionNumber As Byte
		Public LastSectionNumber As Byte
		Public ProtocolDiscriminator As Byte
		Public DsmccType As Byte
		Public MessageId As Short
		Public TransactionId As Integer
		Public Reserved As Byte
		Public AdaptationLength As Byte
		Public MessageLength As Short
		Public RemainingData As Byte
		' Must be marshalled manually
	End Structure

	<StructLayout(LayoutKind.Sequential, Pack := 4)> _
	Public Structure MpegRqstPacket
		Public dwLength As Integer
		<MarshalAs(UnmanagedType.LPStruct)> _
		Public pSection As Section
	End Structure

	<StructLayout(LayoutKind.Sequential, Pack := 1)> _
	Public Structure MpegDate
		Public [Date] As Byte
		Public Month As Byte
		Public Year As Short

		Public Function ToDateTime() As DateTime
			Return New DateTime(Me.Year, Me.Month, Me.[Date])
		End Function
	End Structure

	<StructLayout(LayoutKind.Sequential, Pack := 1)> _
	Public Class MpegServiceRequest
		Public Type As MPEGRequestType
		Public Context As MPEGContext
		Public Pid As Short
		Public TableId As Byte
		Public Filter As MPEG2Filter
		Public Flags As Integer
	End Class

	<StructLayout(LayoutKind.Sequential, Pack := 2)> _
	Public Class MpegServiceResponse
		Public IPAddress As Integer
		Public Port As Short
	End Class

	<StructLayout(LayoutKind.Sequential, Pack := 1)> _
	Public Structure MpegStreamFilter
		Public wPidValue As Short
		Public dwFilterSize As Integer
		<MarshalAs(UnmanagedType.Bool)> _
		Public fCrcEnabled As Boolean
		<MarshalAs(UnmanagedType.ByValArray, ArraySubType := UnmanagedType.U1, SizeConst := 16)> _
		Public rgchFilter As Byte()
		<MarshalAs(UnmanagedType.ByValArray, ArraySubType := UnmanagedType.U1, SizeConst := 16)> _
		Public rgchMask As Byte()
	End Structure
#End If


    <StructLayout(LayoutKind.Sequential, Pack:=1)> _
    Public Structure MpegDuration
        Public Hours As Byte
        Public Minutes As Byte
        Public Seconds As Byte

        Public Function ToTimeSpan() As TimeSpan
            Return New TimeSpan(Me.Hours, Me.Minutes, Me.Seconds)
        End Function
    End Structure

    <StructLayout(LayoutKind.Sequential, Pack:=1)> _
    Public Structure MpegDateAndTime
        'public MpegDate D;
        'public MpegTime T;
        ' Marshaling is faster like that...
        Public [Date] As Byte
        Public Month As Byte
        Public Year As Short
        Public Hours As Byte
        Public Minutes As Byte
        Public Seconds As Byte

        Public Function ToDateTime() As DateTime
            Return New DateTime(Me.Year, Me.Month, Me.[Date], Me.Hours, Me.Minutes, Me.Seconds)
        End Function
    End Structure

    <StructLayout(LayoutKind.Sequential, Pack:=1)> _
    Public Class DsmccElement
        Public pid As Short
        Public bComponentTag As Byte
        Public dwCarouselId As Integer
        Public dwTransactionId As Integer
        Public pNext As DsmccElement
    End Class

    <StructLayout(LayoutKind.Sequential, Pack:=1)> _
    Public Class MpeElement
        Public pid As Short
        Public bComponentTag As Byte
        Public pNext As MpeElement
    End Class

End Namespace
