
Imports System.Runtime.InteropServices

<Flags> _
Public Enum AMInterlace
	None = 0
	IsInterlaced = &H1
	OneFieldPerSample = &H2
	Field1First = &H4
	Unused = &H8
	FieldPatternMask = &H30
	FieldPatField1Only = &H0
	FieldPatField2Only = &H10
	FieldPatBothRegular = &H20
	FieldPatBothIrregular = &H30
	DisplayModeMask = &Hc0
	DisplayModeBobOnly = &H0
	DisplayModeWeaveOnly = &H40
	DisplayModeBobOrWeave = &H80
End Enum

Public Enum AMCopyProtect
	None = 0
	RestrictDuplication = &H1
End Enum

<Flags> _
Public Enum AMControl
	None = 0
	Used = &H1
	PadTo4x3 = &H2
	PadTo16x9 = &H4
End Enum

<StructLayout(LayoutKind.Sequential)> _
Public Class VideoInfoHeader
	Public SrcRect As DsRect
	Public TargetRect As DsRect
	Public BitRate As Integer
	Public BitErrorRate As Integer
	Public AvgTimePerFrame As Long
	Public BmiHeader As BitmapInfoHeader
End Class

<StructLayout(LayoutKind.Sequential)> _
Public Class VideoInfoHeader2
	Public SrcRect As DsRect
	Public TargetRect As DsRect
	Public BitRate As Integer
	Public BitErrorRate As Integer
	Public AvgTimePerFrame As Long
	Public InterlaceFlags As AMInterlace
	Public CopyProtectFlags As AMCopyProtect
	Public PictAspectRatioX As Integer
	Public PictAspectRatioY As Integer
	Public ControlFlags As AMControl
	Public Reserved2 As Integer
	Public BmiHeader As BitmapInfoHeader
End Class

<StructLayout(LayoutKind.Sequential, Pack := 2)> _
Public Class WaveFormatEx
	Public wFormatTag As Short
	Public nChannels As Short
	Public nSamplesPerSec As Integer
	Public nAvgBytesPerSec As Integer
	Public nBlockAlign As Short
	Public wBitsPerSample As Short
	Public cbSize As Short
End Class


<ComImport, System.Security.SuppressUnmanagedCodeSecurity, Guid("6B652FFF-11FE-4fce-92AD-0266B5D7C78F"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)> _
Public Interface ISampleGrabber
	<PreserveSig> _
	Function SetOneShot(<[In], MarshalAs(UnmanagedType.Bool)> OneShot As Boolean) As Integer

	<PreserveSig> _
	Function SetMediaType(<[In], MarshalAs(UnmanagedType.LPStruct)> pmt As AMMediaType) As Integer

	<PreserveSig> _
	Function GetConnectedMediaType(<Out, MarshalAs(UnmanagedType.LPStruct)> pmt As AMMediaType) As Integer

	<PreserveSig> _
	Function SetBufferSamples(<[In], MarshalAs(UnmanagedType.Bool)> BufferThem As Boolean) As Integer

	<PreserveSig> _
	Function GetCurrentBuffer(ByRef pBufferSize As Integer, pBuffer As IntPtr) As Integer

	<PreserveSig> _
	Function GetCurrentSample(ByRef ppSample As IMediaSample) As Integer

	<PreserveSig> _
	Function SetCallback(pCallback As ISampleGrabberCB, WhichMethodToCallback As Integer) As Integer
End Interface

<ComImport, System.Security.SuppressUnmanagedCodeSecurity, Guid("0579154A-2B53-4994-B0D0-E773148EFF85"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)> _
Public Interface ISampleGrabberCB

    ' When called, caller must release pSample
	<PreserveSig> _
	Function SampleCB(SampleTime As Double, pSample As IMediaSample) As Integer

	<PreserveSig> _
	Function BufferCB(SampleTime As Double, pBuffer As IntPtr, BufferLen As Integer) As Integer
End Interface

