namespace Italbytz.ML.UCIMLR;

public static class Data
{
    private static IDataset? _iris;

    public static IDataset Iris => _iris ??= new IrisDataset();
}