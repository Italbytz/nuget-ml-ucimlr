using Microsoft.ML;

namespace Italbytz.ML.UCIMLR;

/// <summary>
///     Provides methods for loading and saving datasets for use with ML.NET.
/// </summary>
public static class StaticData
{
    private static Stream GetStream(DatasetEnum datasetEnum)
    {
        var assembly = typeof(StaticData).Assembly;
        var resourceName = datasetEnum switch
        {
            DatasetEnum.HeartDisease =>
                "Italbytz.ML.UCIMLR.Data.Heart_Disease.csv",
            DatasetEnum.Iris => "Italbytz.ML.UCIMLR.Data.Iris.csv",
            DatasetEnum.WineQuality =>
                "Italbytz.ML.UCIMLR.Data.Wine_Quality.csv",
            _ => null
        };
        var stream = assembly.GetManifestResourceStream(resourceName ??
            throw new InvalidOperationException(
                "Resource for chosen dataset not found"));
        if (stream == null)
            throw new InvalidOperationException(
                $"Resource '{resourceName}' not found.");
        return stream;
    }

    /// <summary>
    ///     Loads a dataset as an <see cref="IDataView" /> for use with ML.NET.
    /// </summary>
    /// <param name="datasetEnum">The dataset to load.</param>
    /// <returns>An <see cref="IDataView" /> containing the loaded data.</returns>
    /// <exception cref="InvalidOperationException">
    ///     Thrown if the resource for the chosen dataset is not found or if loading
    ///     the data fails.
    /// </exception>
    public static IDataView Load(DatasetEnum datasetEnum)
    {
        var stream = GetStream(datasetEnum);
        var tempFile = Path.GetTempFileName();
        using var fileStream = File.Create(tempFile);
        stream?.CopyTo(fileStream);
        fileStream.Flush();
        fileStream.Close();
        var mlContext = new MLContext();
        var data = datasetEnum switch
        {
            DatasetEnum.HeartDisease => mlContext.Data
                .LoadFromTextFile<HeartDiseaseModelInput>(tempFile, ',', true),
            DatasetEnum.Iris => mlContext.Data.LoadFromTextFile<IrisModelInput>(
                tempFile, ',', true),
            DatasetEnum.WineQuality => mlContext.Data
                .LoadFromTextFile<WineQualityModelInput>(tempFile, ',', true),
            _ => null
        };

        return data ??
               throw new InvalidOperationException("Failed to load data");
    }

    /// <summary>
    ///     Saves the specified dataset as a CSV file to the given file path.
    /// </summary>
    /// <param name="datasetEnum">The dataset to save.</param>
    /// <param name="filePath">The file path where the CSV will be saved.</param>
    public static void SaveAsCsv(DatasetEnum datasetEnum, string filePath)
    {
        var stream = GetStream(datasetEnum);
        using var fileStream = File.Create(filePath);
        stream?.CopyTo(fileStream);
        fileStream.Flush();
        fileStream.Close();
    }
}