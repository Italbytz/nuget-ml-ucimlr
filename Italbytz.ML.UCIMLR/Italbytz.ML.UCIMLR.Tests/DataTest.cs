using System.IO;
using Italbytz.ML.UCIMLR;
using JetBrains.Annotations;
using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace Italbytz.ML.UCIMLR.Tests;

[TestClass]
[TestSubject(typeof(Data))]
public class DataTest
{

    [TestMethod]
    public void TestLoadIris()
    {
        const Dataset dataset = Dataset.Iris;
        var data = Data.Load(dataset);
        Assert.IsNotNull(data);
    }
    
    [TestMethod]
    public void TestLoadHeartDisease()
    {
        const Dataset dataset = Dataset.HeartDisease;
        var data = Data.Load(dataset);
        Assert.IsNotNull(data);
    }
    
    [TestMethod]
    public void TestLoadWineQuality()
    {
        const Dataset dataset = Dataset.WineQuality;
        var data = Data.Load(dataset);
        Assert.IsNotNull(data);
    }
    
    [TestMethod]
    public void TestSaveAsCsv()
    {
        const Dataset dataset = Dataset.Iris;
        var filePath = Path.Combine(Path.GetTempPath(), "Iris.csv");
        Data.SaveAsCsv(dataset, filePath);
        Assert.IsTrue(File.Exists(filePath));
        File.Delete(filePath);
    }
}