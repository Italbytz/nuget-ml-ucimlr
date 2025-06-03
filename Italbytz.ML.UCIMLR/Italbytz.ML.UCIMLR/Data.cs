namespace Italbytz.ML.UCIMLR;

public static class Data
{
    private static IDataset? _iris;
    private static IDataset? _breastCancerWisconsinDiagnostic;
    private static IDataset? _wineQuality;
    private static IDataset? _heartDisease;
    private static IDataset? _heartDiseaseBinary;

    public static IDataset Iris => _iris ??= new IrisDataset();

    public static IDataset BreastCancerWisconsinDiagnostic =>
        _breastCancerWisconsinDiagnostic ??=
            new BreastCancerWisconsinDiagnosticDataset();

    public static IDataset WineQuality =>
        _wineQuality ??= new WineQualityDataset();

    public static IDataset HeartDisease =>
        _heartDisease ??= new HeartDiseaseDataset();

    public static IDataset HeartDiseaseBinary =>
        _heartDiseaseBinary ??= new HeartDiseaseBinaryDataset();
}