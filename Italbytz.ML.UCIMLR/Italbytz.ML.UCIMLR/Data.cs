namespace Italbytz.ML.UCIMLR;

public static class Data
{
    private static IDataset iris;

    public static IDataset Iris => iris ??= new IrisDataset();
}