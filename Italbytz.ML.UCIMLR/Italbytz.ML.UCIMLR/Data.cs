using Microsoft.ML;

namespace Italbytz.ML.UCIMLR;

public static class Data
{
    private static Stream GetStream(Dataset dataset)
    {
        var assembly = typeof(Data).Assembly;
        var resourceName = dataset switch
        {
            Dataset.HeartDisease => "Italbytz.ML.UCIMLR.Data.Heart_Disease.csv",
            Dataset.Iris => "Italbytz.ML.UCIMLR.Data.Iris.csv",
            Dataset.WineQuality => "Italbytz.ML.UCIMLR.Data.Wine_Quality.csv",
            _ => null
        };
        var stream = assembly.GetManifestResourceStream(resourceName ?? throw new InvalidOperationException("Resource for chosen dataset not found"));
        if (stream == null)
        {
            throw new InvalidOperationException($"Resource '{resourceName}' not found.");
        }
        return stream;
    }
    
    public static IDataView Load(Dataset dataset)
    {
        var stream = GetStream(dataset);
        var tempFile = Path.GetTempFileName();
        using var fileStream = File.Create(tempFile);
        stream?.CopyTo(fileStream);
        fileStream.Flush();
        fileStream.Close();
        var mlContext = new MLContext();
        var data = dataset switch 
        {
            Dataset.HeartDisease => mlContext.Data.LoadFromTextFile<HeartDiseaseModelInput>(tempFile, separatorChar: ',', hasHeader: true),
            Dataset.Iris => mlContext.Data.LoadFromTextFile<IrisModelInput>(tempFile, separatorChar: ',', hasHeader: true),
            Dataset.WineQuality => mlContext.Data.LoadFromTextFile<WineQualityModelInput>(tempFile, separatorChar: ',', hasHeader: true),
            _ => null
        };
            
        return data ?? throw new InvalidOperationException("Failed to load data");
    }
    
    public static void SaveAsCsv(Dataset dataset, string filePath)
    {
        var stream = GetStream(dataset);
        using var fileStream = File.Create(filePath);
        stream?.CopyTo(fileStream);
        fileStream.Flush();
        fileStream.Close();
    }
}