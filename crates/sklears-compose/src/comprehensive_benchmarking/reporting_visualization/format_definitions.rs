//! Comprehensive format definitions for export engines
//!
//! This module provides extensive format support for data visualization exports,
//! including image, vector, document, data, web, presentation, and interactive formats.

use std::collections::HashMap;
use serde::{Serialize, Deserialize};
use chrono::{DateTime, Utc, Duration};

/// Export format enumeration with comprehensive format support
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ExportFormat {
    /// Image formats
    Image(ImageFormat),
    /// Vector formats
    Vector(VectorFormat),
    /// Document formats
    Document(DocumentFormat),
    /// Data formats
    Data(DataFormat),
    /// Web formats
    Web(WebFormat),
    /// Presentation formats
    Presentation(PresentationFormat),
    /// Interactive formats
    Interactive(InteractiveFormat),
    /// Custom format
    Custom(String),
}

/// Image format types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ImageFormat {
    /// PNG format
    PNG(PngOptions),
    /// JPEG format
    JPEG(JpegOptions),
    /// WebP format
    WebP(WebPOptions),
    /// AVIF format
    AVIF(AvifOptions),
    /// HEIF format
    HEIF(HeifOptions),
    /// TIFF format
    TIFF(TiffOptions),
    /// BMP format
    BMP(BmpOptions),
    /// GIF format
    GIF(GifOptions),
    /// ICO format
    ICO(IcoOptions),
    /// Custom image format
    Custom(String),
}

/// PNG format options
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PngOptions {
    /// Compression level (0-9)
    pub compression_level: u8,
    /// Color type
    pub color_type: PngColorType,
    /// Bit depth
    pub bit_depth: PngBitDepth,
    /// Interlacing
    pub interlacing: bool,
    /// Metadata preservation
    pub preserve_metadata: bool,
}

/// PNG color types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PngColorType {
    /// Grayscale
    Grayscale,
    /// RGB
    RGB,
    /// Palette
    Palette,
    /// Grayscale with alpha
    GrayscaleAlpha,
    /// RGB with alpha
    RGBA,
}

/// PNG bit depths
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PngBitDepth {
    /// 1 bit
    Bit1,
    /// 2 bits
    Bit2,
    /// 4 bits
    Bit4,
    /// 8 bits
    Bit8,
    /// 16 bits
    Bit16,
}

/// JPEG format options
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JpegOptions {
    /// Quality (1-100)
    pub quality: u8,
    /// Progressive encoding
    pub progressive: bool,
    /// Chroma subsampling
    pub chroma_subsampling: ChromaSubsampling,
    /// Optimization
    pub optimization: bool,
    /// Metadata preservation
    pub preserve_metadata: bool,
}

/// Chroma subsampling options
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ChromaSubsampling {
    /// 4:4:4 (no subsampling)
    Full,
    /// 4:2:2 (horizontal subsampling)
    Horizontal,
    /// 4:2:0 (both directions subsampling)
    Both,
    /// 4:1:1 (aggressive subsampling)
    Aggressive,
}

/// WebP format options
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WebPOptions {
    /// Lossy compression quality
    pub lossy_quality: Option<f32>,
    /// Lossless compression
    pub lossless: bool,
    /// Animation support
    pub animation: bool,
    /// Alpha channel
    pub alpha_channel: bool,
    /// Metadata preservation
    pub preserve_metadata: bool,
}

/// AVIF format options
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AvifOptions {
    /// Quality (0-100)
    pub quality: u8,
    /// Speed vs quality trade-off
    pub speed: AvifSpeed,
    /// Chroma subsampling
    pub chroma_subsampling: ChromaSubsampling,
    /// Bit depth
    pub bit_depth: u8,
    /// Animation support
    pub animation: bool,
}

/// AVIF encoding speed
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AvifSpeed {
    /// Slowest encoding, best quality
    Slowest,
    /// Slow encoding
    Slow,
    /// Medium speed
    Medium,
    /// Fast encoding
    Fast,
    /// Fastest encoding, lower quality
    Fastest,
}

/// HEIF format options
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HeifOptions {
    /// Quality (0-100)
    pub quality: u8,
    /// Lossless compression
    pub lossless: bool,
    /// Sequence support
    pub sequence: bool,
    /// Metadata preservation
    pub preserve_metadata: bool,
}

/// TIFF format options
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TiffOptions {
    /// Compression type
    pub compression: TiffCompression,
    /// Color space
    pub color_space: TiffColorSpace,
    /// Bit depth
    pub bit_depth: u8,
    /// Multi-page support
    pub multi_page: bool,
    /// Metadata preservation
    pub preserve_metadata: bool,
}

/// TIFF compression types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TiffCompression {
    /// No compression
    None,
    /// LZW compression
    LZW,
    /// ZIP compression
    ZIP,
    /// JPEG compression
    JPEG,
    /// CCITT compression
    CCITT,
}

/// TIFF color spaces
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TiffColorSpace {
    /// RGB color space
    RGB,
    /// CMYK color space
    CMYK,
    /// LAB color space
    LAB,
    /// Grayscale
    Grayscale,
}

/// BMP format options
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BmpOptions {
    /// BMP version
    pub version: BmpVersion,
    /// Compression
    pub compression: bool,
    /// Color depth
    pub color_depth: u8,
}

/// BMP versions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum BmpVersion {
    /// Version 3
    V3,
    /// Version 4
    V4,
    /// Version 5
    V5,
}

/// GIF format options
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GifOptions {
    /// Animation support
    pub animation: bool,
    /// Loop count
    pub loop_count: Option<u16>,
    /// Color quantization
    pub color_quantization: ColorQuantization,
    /// Dithering
    pub dithering: bool,
    /// Transparency
    pub transparency: bool,
}

/// Color quantization methods
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ColorQuantization {
    /// Median cut
    MedianCut,
    /// Octree
    Octree,
    /// K-means
    KMeans,
    /// Uniform quantization
    Uniform,
    /// Custom quantization
    Custom(String),
}

/// ICO format options
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IcoOptions {
    /// Icon sizes
    pub sizes: Vec<(u32, u32)>,
    /// Color depths
    pub color_depths: Vec<u8>,
    /// PNG compression
    pub png_compression: bool,
}

/// Vector format types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum VectorFormat {
    /// SVG format
    SVG(SvgOptions),
    /// EPS format
    EPS(EpsOptions),
    /// PDF vector format
    PDFVector(PdfVectorOptions),
    /// AI format
    AI(AiOptions),
    /// EMF format
    EMF(EmfOptions),
    /// WMF format
    WMF(WmfOptions),
    /// Custom vector format
    Custom(String),
}

/// SVG format options
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SvgOptions {
    /// Optimization level
    pub optimization_level: SvgOptimizationLevel,
    /// Decimal precision
    pub decimal_precision: u8,
    /// Include metadata
    pub include_metadata: bool,
    /// Minification
    pub minification: bool,
    /// Pretty printing
    pub pretty_print: bool,
    /// Font embedding
    pub font_embedding: FontEmbedding,
}

/// SVG optimization levels
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SvgOptimizationLevel {
    /// No optimization
    None,
    /// Basic optimization
    Basic,
    /// Advanced optimization
    Advanced,
    /// Aggressive optimization
    Aggressive,
}

/// Font embedding options
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum FontEmbedding {
    /// No font embedding
    None,
    /// Embed used fonts
    UsedFonts,
    /// Embed all fonts
    AllFonts,
    /// Convert to paths
    ConvertToPaths,
}

/// EPS format options
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EpsOptions {
    /// EPS version
    pub version: EpsVersion,
    /// Bounding box
    pub bounding_box: bool,
    /// Preview image
    pub preview_image: bool,
    /// Font embedding
    pub font_embedding: FontEmbedding,
}

/// EPS versions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum EpsVersion {
    /// Level 1
    Level1,
    /// Level 2
    Level2,
    /// Level 3
    Level3,
}

/// PDF vector format options
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PdfVectorOptions {
    /// PDF version
    pub pdf_version: PdfVersion,
    /// Compression
    pub compression: PdfCompression,
    /// Font embedding
    pub font_embedding: FontEmbedding,
    /// Color profile
    pub color_profile: Option<String>,
    /// Metadata
    pub metadata: PdfMetadata,
}

/// PDF versions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PdfVersion {
    /// Version 1.4
    V1_4,
    /// Version 1.5
    V1_5,
    /// Version 1.6
    V1_6,
    /// Version 1.7
    V1_7,
    /// Version 2.0
    V2_0,
}

/// PDF compression options
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PdfCompression {
    /// Text compression
    pub text_compression: bool,
    /// Image compression
    pub image_compression: ImageCompressionLevel,
    /// Vector compression
    pub vector_compression: bool,
}

/// Image compression levels
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ImageCompressionLevel {
    /// No compression
    None,
    /// Low compression
    Low,
    /// Medium compression
    Medium,
    /// High compression
    High,
    /// Maximum compression
    Maximum,
}

/// PDF metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PdfMetadata {
    /// Document title
    pub title: Option<String>,
    /// Document author
    pub author: Option<String>,
    /// Document subject
    pub subject: Option<String>,
    /// Document keywords
    pub keywords: Vec<String>,
    /// Creation date
    pub creation_date: Option<DateTime<Utc>>,
    /// Modification date
    pub modification_date: Option<DateTime<Utc>>,
}

/// AI format options
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AiOptions {
    /// AI version
    pub version: AiVersion,
    /// Compatibility mode
    pub compatibility_mode: bool,
    /// Font embedding
    pub font_embedding: FontEmbedding,
    /// Compression
    pub compression: bool,
}

/// AI format versions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AiVersion {
    /// CS3
    CS3,
    /// CS4
    CS4,
    /// CS5
    CS5,
    /// CS6
    CS6,
    /// CC
    CC,
}

/// EMF format options
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmfOptions {
    /// Enhanced features
    pub enhanced_features: bool,
    /// Text as curves
    pub text_as_curves: bool,
    /// Image embedding
    pub image_embedding: bool,
}

/// WMF format options
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WmfOptions {
    /// Placeable format
    pub placeable: bool,
    /// Text handling
    pub text_handling: WmfTextHandling,
}

/// WMF text handling options
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum WmfTextHandling {
    /// Keep as text
    KeepText,
    /// Convert to curves
    ConvertToCurves,
    /// Rasterize text
    Rasterize,
}

/// Document format types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DocumentFormat {
    /// PDF document
    PDF(PdfOptions),
    /// Word document
    DOCX(DocxOptions),
    /// PowerPoint presentation
    PPTX(PptxOptions),
    /// RTF document
    RTF(RtfOptions),
    /// LaTeX document
    LaTeX(LatexOptions),
    /// HTML document
    HTML(HtmlOptions),
    /// Markdown document
    Markdown(MarkdownOptions),
    /// Custom document format
    Custom(String),
}

/// PDF document options
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PdfOptions {
    /// Page size
    pub page_size: PageSize,
    /// Orientation
    pub orientation: PageOrientation,
    /// Margins
    pub margins: PageMargins,
    /// Quality settings
    pub quality: PdfQuality,
    /// Security settings
    pub security: PdfSecurity,
    /// Accessibility
    pub accessibility: PdfAccessibility,
}

/// Page size options
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PageSize {
    /// A4 size
    A4,
    /// A3 size
    A3,
    /// A5 size
    A5,
    /// Letter size
    Letter,
    /// Legal size
    Legal,
    /// Tabloid size
    Tabloid,
    /// Custom size
    Custom(f64, f64),
}

/// Page orientation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PageOrientation {
    /// Portrait orientation
    Portrait,
    /// Landscape orientation
    Landscape,
}

/// Page margins
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PageMargins {
    /// Top margin
    pub top: f64,
    /// Right margin
    pub right: f64,
    /// Bottom margin
    pub bottom: f64,
    /// Left margin
    pub left: f64,
}

/// PDF quality settings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PdfQuality {
    /// Image quality
    pub image_quality: ImageQuality,
    /// Text quality
    pub text_quality: TextQuality,
    /// Vector quality
    pub vector_quality: VectorQuality,
}

/// Image quality settings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImageQuality {
    /// DPI setting
    pub dpi: u32,
    /// Compression level
    pub compression: ImageCompressionLevel,
    /// Color space
    pub color_space: ImageColorSpace,
}

/// Image color spaces
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ImageColorSpace {
    /// RGB color space
    RGB,
    /// CMYK color space
    CMYK,
    /// Grayscale
    Grayscale,
    /// Lab color space
    Lab,
}

/// Text quality settings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TextQuality {
    /// Font embedding
    pub font_embedding: FontEmbedding,
    /// Subset fonts
    pub subset_fonts: bool,
    /// Anti-aliasing
    pub anti_aliasing: bool,
}

/// Vector quality settings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VectorQuality {
    /// Precision
    pub precision: VectorPrecision,
    /// Optimization
    pub optimization: bool,
    /// Simplification
    pub simplification: bool,
}

/// Vector precision levels
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum VectorPrecision {
    /// Low precision
    Low,
    /// Medium precision
    Medium,
    /// High precision
    High,
    /// Maximum precision
    Maximum,
}

/// PDF security settings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PdfSecurity {
    /// Encryption enabled
    pub encryption: bool,
    /// User password
    pub user_password: Option<String>,
    /// Owner password
    pub owner_password: Option<String>,
    /// Permissions
    pub permissions: PdfPermissions,
}

/// PDF permissions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PdfPermissions {
    /// Allow printing
    pub allow_printing: bool,
    /// Allow copying
    pub allow_copying: bool,
    /// Allow modifications
    pub allow_modifications: bool,
    /// Allow annotations
    pub allow_annotations: bool,
    /// Allow form filling
    pub allow_form_filling: bool,
    /// Allow assembly
    pub allow_assembly: bool,
    /// Allow degraded printing
    pub allow_degraded_printing: bool,
}

/// PDF accessibility settings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PdfAccessibility {
    /// Tagged PDF
    pub tagged_pdf: bool,
    /// Alt text for images
    pub alt_text: bool,
    /// Logical reading order
    pub logical_reading_order: bool,
    /// Language specification
    pub language: Option<String>,
}

/// Data format types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DataFormat {
    /// CSV format
    CSV(CsvOptions),
    /// JSON format
    JSON(JsonOptions),
    /// XML format
    XML(XmlOptions),
    /// Excel format
    Excel(ExcelOptions),
    /// Parquet format
    Parquet(ParquetOptions),
    /// Custom data format
    Custom(String),
}

/// CSV format options
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CsvOptions {
    /// Delimiter character
    pub delimiter: char,
    /// Quote character
    pub quote_char: char,
    /// Escape character
    pub escape_char: Option<char>,
    /// Include headers
    pub headers: bool,
    /// Encoding
    pub encoding: TextEncoding,
}

/// JSON format options
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JsonOptions {
    /// Pretty printing
    pub pretty_print: bool,
    /// Compact format
    pub compact: bool,
    /// Encoding
    pub encoding: TextEncoding,
    /// Include schema
    pub include_schema: bool,
}

/// XML format options
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct XmlOptions {
    /// XML version
    pub xml_version: XmlVersion,
    /// Encoding
    pub encoding: TextEncoding,
    /// Pretty printing
    pub pretty_print: bool,
    /// Indentation
    pub indentation: XmlIndentation,
    /// Namespace handling
    pub namespace_handling: NamespaceHandling,
    /// Schema validation
    pub schema_validation: SchemaValidation,
}

/// XML versions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum XmlVersion {
    /// Version 1.0
    V1_0,
    /// Version 1.1
    V1_1,
}

/// XML indentation options
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum XmlIndentation {
    /// No indentation
    None,
    /// Spaces
    Spaces(u8),
    /// Tabs
    Tabs,
}

/// Namespace handling options
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NamespaceHandling {
    /// Include namespaces
    pub include_namespaces: bool,
    /// Default namespace
    pub default_namespace: Option<String>,
    /// Namespace prefixes
    pub namespace_prefixes: HashMap<String, String>,
}

/// Schema validation options
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SchemaValidation {
    /// Enable validation
    pub enabled: bool,
    /// Schema file
    pub schema_file: Option<String>,
    /// Validation level
    pub validation_level: ValidationLevel,
}

/// Validation levels
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ValidationLevel {
    /// Strict validation
    Strict,
    /// Lenient validation
    Lenient,
    /// Warning only
    Warning,
}

/// Excel format options
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExcelOptions {
    /// Excel version
    pub excel_version: ExcelVersion,
    /// Worksheet options
    pub worksheet_options: WorksheetOptions,
    /// Formatting options
    pub formatting_options: ExcelFormatting,
    /// Charts inclusion
    pub include_charts: bool,
}

/// Excel versions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ExcelVersion {
    /// Excel 2007
    Excel2007,
    /// Excel 2010
    Excel2010,
    /// Excel 2013
    Excel2013,
    /// Excel 2016
    Excel2016,
    /// Excel 365
    Excel365,
}

/// Worksheet options
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorksheetOptions {
    /// Sheet names
    pub sheet_names: Vec<String>,
    /// Include headers
    pub include_headers: bool,
    /// Auto-fit columns
    pub auto_fit_columns: bool,
    /// Freeze panes
    pub freeze_panes: Option<(u32, u32)>,
}

/// Excel formatting options
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExcelFormatting {
    /// Number formats
    pub number_formats: HashMap<String, String>,
    /// Cell styles
    pub cell_styles: Vec<CellStyle>,
    /// Conditional formatting
    pub conditional_formatting: bool,
}

/// Cell style definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CellStyle {
    /// Style name
    pub name: String,
    /// Font settings
    pub font: FontSettings,
    /// Border settings
    pub border: BorderSettings,
    /// Fill settings
    pub fill: FillSettings,
    /// Alignment
    pub alignment: CellAlignment,
}

/// Font settings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FontSettings {
    /// Font name
    pub name: String,
    /// Font size
    pub size: f64,
    /// Font weight
    pub weight: FontWeight,
    /// Font style
    pub style: FontStyle,
    /// Font color
    pub color: String,
}

/// Font weight options
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum FontWeight {
    /// Normal weight
    Normal,
    /// Bold weight
    Bold,
    /// Light weight
    Light,
    /// Custom weight
    Custom(u16),
}

/// Font style options
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum FontStyle {
    /// Normal style
    Normal,
    /// Italic style
    Italic,
    /// Oblique style
    Oblique,
}

/// Border settings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BorderSettings {
    /// Top border
    pub top: BorderStyle,
    /// Right border
    pub right: BorderStyle,
    /// Bottom border
    pub bottom: BorderStyle,
    /// Left border
    pub left: BorderStyle,
}

/// Border style definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BorderStyle {
    /// Border width
    pub width: f64,
    /// Border color
    pub color: String,
    /// Border line style
    pub line_style: LineStyle,
}

/// Line style options
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum LineStyle {
    /// Solid line
    Solid,
    /// Dashed line
    Dashed,
    /// Dotted line
    Dotted,
    /// Double line
    Double,
    /// Custom line style
    Custom(String),
}

/// Fill settings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FillSettings {
    /// Fill type
    pub fill_type: FillType,
    /// Primary color
    pub primary_color: String,
    /// Secondary color
    pub secondary_color: Option<String>,
    /// Pattern
    pub pattern: Option<FillPattern>,
}

/// Fill type options
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum FillType {
    /// Solid fill
    Solid,
    /// Gradient fill
    Gradient,
    /// Pattern fill
    Pattern,
    /// No fill
    None,
}

/// Fill pattern options
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum FillPattern {
    /// Diagonal stripes
    DiagonalStripes,
    /// Horizontal stripes
    HorizontalStripes,
    /// Vertical stripes
    VerticalStripes,
    /// Dots
    Dots,
    /// Custom pattern
    Custom(String),
}

/// Cell alignment
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CellAlignment {
    /// Horizontal alignment
    pub horizontal: HorizontalAlignment,
    /// Vertical alignment
    pub vertical: VerticalAlignment,
    /// Text wrapping
    pub wrap_text: bool,
    /// Text rotation
    pub rotation: Option<f64>,
}

/// Horizontal alignment options
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum HorizontalAlignment {
    /// Left alignment
    Left,
    /// Center alignment
    Center,
    /// Right alignment
    Right,
    /// Justify alignment
    Justify,
}

/// Vertical alignment options
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum VerticalAlignment {
    /// Top alignment
    Top,
    /// Middle alignment
    Middle,
    /// Bottom alignment
    Bottom,
}

/// Parquet format options
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParquetOptions {
    /// Compression codec
    pub compression: ParquetCompression,
    /// Row group size
    pub row_group_size: usize,
    /// Page size
    pub page_size: usize,
    /// Enable statistics
    pub enable_statistics: bool,
}

/// Parquet compression options
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ParquetCompression {
    /// No compression
    None,
    /// Snappy compression
    Snappy,
    /// Gzip compression
    Gzip,
    /// LZO compression
    LZO,
    /// Brotli compression
    Brotli,
}

/// Web format types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum WebFormat {
    /// HTML format
    HTML(HtmlOptions),
    /// CSS format
    CSS(CssOptions),
    /// JavaScript format
    JavaScript(JavaScriptOptions),
    /// Custom web format
    Custom(String),
}

/// HTML format options
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HtmlOptions {
    /// HTML version
    pub html_version: HtmlVersion,
    /// Include CSS
    pub include_css: bool,
    /// Include JavaScript
    pub include_javascript: bool,
    /// Responsive design
    pub responsive: bool,
    /// Accessibility features
    pub accessibility: HtmlAccessibility,
}

/// HTML versions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum HtmlVersion {
    /// HTML 4.01
    HTML4_01,
    /// XHTML 1.0
    XHTML1_0,
    /// HTML5
    HTML5,
}

/// HTML accessibility options
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HtmlAccessibility {
    /// Include alt text
    pub alt_text: bool,
    /// ARIA labels
    pub aria_labels: bool,
    /// Semantic markup
    pub semantic_markup: bool,
    /// Screen reader support
    pub screen_reader_support: bool,
}

/// CSS format options
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CssOptions {
    /// CSS version
    pub css_version: CssVersion,
    /// Minification
    pub minification: bool,
    /// Source maps
    pub source_maps: bool,
    /// Vendor prefixes
    pub vendor_prefixes: bool,
}

/// CSS versions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CssVersion {
    /// CSS 2.1
    CSS2_1,
    /// CSS 3
    CSS3,
    /// CSS 4
    CSS4,
}

/// JavaScript format options
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JavaScriptOptions {
    /// ECMAScript version
    pub ecmascript_version: EcmaScriptVersion,
    /// Minification
    pub minification: bool,
    /// Source maps
    pub source_maps: bool,
    /// Module format
    pub module_format: ModuleFormat,
}

/// ECMAScript versions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum EcmaScriptVersion {
    /// ES5
    ES5,
    /// ES6 (ES2015)
    ES6,
    /// ES2017
    ES2017,
    /// ES2018
    ES2018,
    /// ES2019
    ES2019,
    /// ES2020
    ES2020,
}

/// Module format options
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ModuleFormat {
    /// CommonJS
    CommonJS,
    /// ES Modules
    ESModules,
    /// AMD
    AMD,
    /// UMD
    UMD,
}

/// Presentation format types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PresentationFormat {
    /// PowerPoint
    PowerPoint(PowerPointOptions),
    /// PDF presentation
    PDFPresentation(PdfPresentationOptions),
    /// HTML presentation
    HTMLPresentation(HtmlPresentationOptions),
    /// Custom presentation format
    Custom(String),
}

/// PowerPoint options
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PowerPointOptions {
    /// Slide size
    pub slide_size: SlideSize,
    /// Template
    pub template: Option<String>,
    /// Animation export
    pub animation_export: bool,
    /// Notes inclusion
    pub include_notes: bool,
}

/// Slide size options
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SlideSize {
    /// Standard 4:3
    Standard,
    /// Widescreen 16:9
    Widescreen,
    /// Custom size
    Custom(f64, f64),
}

/// PDF presentation options
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PdfPresentationOptions {
    /// PDF options
    pub pdf_options: PdfOptions,
    /// Presentation mode
    pub presentation_mode: bool,
    /// Full screen
    pub full_screen: bool,
    /// Transitions
    pub transitions: bool,
}

/// HTML presentation options
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HtmlPresentationOptions {
    /// HTML options
    pub html_options: HtmlOptions,
    /// Presentation framework
    pub framework: PresentationFramework,
    /// Theme
    pub theme: Option<String>,
    /// Interactive features
    pub interactive_features: bool,
}

/// Presentation framework options
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PresentationFramework {
    /// Reveal.js
    RevealJS,
    /// Impress.js
    ImpressJS,
    /// Deck.js
    DeckJS,
    /// Custom framework
    Custom(String),
}

/// Interactive format types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum InteractiveFormat {
    /// WebGL format
    WebGL(WebGlOptions),
    /// Canvas format
    Canvas(CanvasOptions),
    /// D3 format
    D3(D3Options),
    /// Custom interactive format
    Custom(String),
}

/// WebGL format options
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WebGlOptions {
    /// WebGL version
    pub webgl_version: WebGlVersion,
    /// Anti-aliasing
    pub anti_aliasing: bool,
    /// Depth buffer
    pub depth_buffer: bool,
    /// Stencil buffer
    pub stencil_buffer: bool,
}

/// WebGL versions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum WebGlVersion {
    /// WebGL 1.0
    WebGL1,
    /// WebGL 2.0
    WebGL2,
}

/// Canvas format options
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CanvasOptions {
    /// Canvas type
    pub canvas_type: CanvasType,
    /// High DPI support
    pub high_dpi: bool,
    /// Image smoothing
    pub image_smoothing: bool,
}

/// Canvas type options
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CanvasType {
    /// 2D canvas
    Canvas2D,
    /// OffscreenCanvas
    OffscreenCanvas,
    /// ImageBitmap
    ImageBitmap,
}

/// D3 format options
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct D3Options {
    /// D3 version
    pub d3_version: D3Version,
    /// Animation support
    pub animation: bool,
    /// Interaction support
    pub interaction: bool,
    /// Data binding
    pub data_binding: bool,
}

/// D3 versions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum D3Version {
    /// Version 3
    V3,
    /// Version 4
    V4,
    /// Version 5
    V5,
    /// Version 6
    V6,
    /// Version 7
    V7,
}

/// Text encoding options
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TextEncoding {
    /// UTF-8
    UTF8,
    /// UTF-16
    UTF16,
    /// UTF-32
    UTF32,
    /// ASCII
    ASCII,
    /// ISO-8859-1
    ISO8859_1,
    /// Windows-1252
    Windows1252,
    /// Custom encoding
    Custom(String),
}

/// RTF document options
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RtfOptions {
    /// RTF version
    pub version: RtfVersion,
    /// Font embedding
    pub font_embedding: bool,
    /// Image embedding
    pub image_embedding: bool,
    /// Compatibility settings
    pub compatibility: RtfCompatibility,
}

/// RTF versions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RtfVersion {
    /// Version 1.0
    V1_0,
    /// Version 1.5
    V1_5,
    /// Version 1.9
    V1_9,
}

/// RTF compatibility settings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RtfCompatibility {
    /// Microsoft Word compatibility
    pub word_compatibility: bool,
    /// LibreOffice compatibility
    pub libreoffice_compatibility: bool,
    /// Google Docs compatibility
    pub google_docs_compatibility: bool,
}

/// LaTeX document options
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LatexOptions {
    /// Document class
    pub document_class: LatexDocumentClass,
    /// Packages
    pub packages: Vec<String>,
    /// Bibliography style
    pub bibliography_style: Option<String>,
    /// Math mode
    pub math_mode: LatexMathMode,
}

/// LaTeX document classes
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum LatexDocumentClass {
    /// Article class
    Article,
    /// Report class
    Report,
    /// Book class
    Book,
    /// Letter class
    Letter,
    /// Custom class
    Custom(String),
}

/// LaTeX math modes
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum LatexMathMode {
    /// Inline math
    Inline,
    /// Display math
    Display,
    /// Both modes
    Both,
}

/// Markdown document options
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MarkdownOptions {
    /// Markdown flavor
    pub flavor: MarkdownFlavor,
    /// Extensions
    pub extensions: Vec<MarkdownExtension>,
    /// HTML output
    pub html_output: bool,
    /// Table of contents
    pub table_of_contents: bool,
}

/// Markdown flavors
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MarkdownFlavor {
    /// CommonMark
    CommonMark,
    /// GitHub Flavored Markdown
    GitHubFlavored,
    /// MultiMarkdown
    MultiMarkdown,
    /// Pandoc Markdown
    Pandoc,
    /// Custom flavor
    Custom(String),
}

/// Markdown extensions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MarkdownExtension {
    /// Tables
    Tables,
    /// Task lists
    TaskLists,
    /// Strikethrough
    Strikethrough,
    /// Footnotes
    Footnotes,
    /// Math support
    Math,
    /// Syntax highlighting
    SyntaxHighlighting,
    /// Custom extension
    Custom(String),
}

/// DOCX document options
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DocxOptions {
    /// Document template
    pub template: Option<String>,
    /// Page setup
    pub page_setup: PageSetup,
    /// Styles
    pub styles: DocxStyles,
    /// Compatibility
    pub compatibility: DocxCompatibility,
}

/// Page setup for documents
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PageSetup {
    /// Page size
    pub page_size: PageSize,
    /// Orientation
    pub orientation: PageOrientation,
    /// Margins
    pub margins: PageMargins,
    /// Header/footer
    pub header_footer: HeaderFooterSettings,
}

/// Header and footer settings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HeaderFooterSettings {
    /// Include header
    pub include_header: bool,
    /// Include footer
    pub include_footer: bool,
    /// Different first page
    pub different_first_page: bool,
    /// Different odd/even pages
    pub different_odd_even: bool,
}

/// DOCX styles
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DocxStyles {
    /// Default font
    pub default_font: String,
    /// Heading styles
    pub heading_styles: Vec<HeadingStyle>,
    /// Paragraph styles
    pub paragraph_styles: Vec<ParagraphStyle>,
    /// Table styles
    pub table_styles: Vec<TableStyle>,
}

/// Heading style definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HeadingStyle {
    /// Level (1-6)
    pub level: u8,
    /// Font name
    pub font_name: String,
    /// Font size
    pub font_size: f64,
    /// Font weight
    pub font_weight: FontWeight,
    /// Color
    pub color: String,
}

/// Paragraph style definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParagraphStyle {
    /// Style name
    pub name: String,
    /// Font settings
    pub font: FontSettings,
    /// Alignment
    pub alignment: TextAlignment,
    /// Line spacing
    pub line_spacing: LineSpacing,
    /// Indentation
    pub indentation: Indentation,
}

/// Text alignment options
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TextAlignment {
    /// Left alignment
    Left,
    /// Center alignment
    Center,
    /// Right alignment
    Right,
    /// Justify alignment
    Justify,
}

/// Line spacing options
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum LineSpacing {
    /// Single spacing
    Single,
    /// Double spacing
    Double,
    /// Custom spacing
    Custom(f64),
}

/// Indentation settings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Indentation {
    /// Left indentation
    pub left: f64,
    /// Right indentation
    pub right: f64,
    /// First line indentation
    pub first_line: f64,
    /// Hanging indentation
    pub hanging: f64,
}

/// Table style definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TableStyle {
    /// Style name
    pub name: String,
    /// Border settings
    pub border: BorderSettings,
    /// Cell padding
    pub cell_padding: f64,
    /// Header row formatting
    pub header_formatting: Option<CellStyle>,
    /// Alternating row colors
    pub alternating_rows: bool,
}

/// DOCX compatibility settings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DocxCompatibility {
    /// Word version
    pub word_version: WordVersion,
    /// Compatibility mode
    pub compatibility_mode: bool,
    /// Legacy features
    pub legacy_features: bool,
}

/// Word versions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum WordVersion {
    /// Word 2010
    Word2010,
    /// Word 2013
    Word2013,
    /// Word 2016
    Word2016,
    /// Word 2019
    Word2019,
    /// Word 365
    Word365,
}

/// PPTX presentation options
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PptxOptions {
    /// Slide size
    pub slide_size: SlideSize,
    /// Template
    pub template: Option<String>,
    /// Presentation settings
    pub presentation_settings: PresentationSettings,
    /// Animation export
    pub animation_export: AnimationExportSettings,
}

/// Presentation settings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PresentationSettings {
    /// Auto advance slides
    pub auto_advance: bool,
    /// Loop presentation
    pub loop_presentation: bool,
    /// Show navigation
    pub show_navigation: bool,
    /// Full screen mode
    pub full_screen: bool,
}

/// Animation export settings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnimationExportSettings {
    /// Export animations
    pub export_animations: bool,
    /// Animation quality
    pub animation_quality: AnimationQuality,
    /// Timing preservation
    pub preserve_timing: bool,
}

/// Animation quality levels
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AnimationQuality {
    /// Low quality
    Low,
    /// Medium quality
    Medium,
    /// High quality
    High,
    /// Original quality
    Original,
}