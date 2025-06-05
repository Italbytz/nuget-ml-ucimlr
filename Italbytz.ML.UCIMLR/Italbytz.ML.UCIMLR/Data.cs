namespace Italbytz.ML.Data;

public static class Data
{
    private static IDataset? _iris;
    private static IDataset? _breastCancerWisconsinDiagnostic;
    private static IDataset? _wineQuality;
    private static IDataset? _heartDisease;
    private static IDataset? _heartDiseaseBinary;
    private static IDataset? _adult;

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

    public static IDataset Adult =>
        _adult ??= new AdultDataset();
}