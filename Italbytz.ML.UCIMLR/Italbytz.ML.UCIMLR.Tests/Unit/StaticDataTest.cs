using System.IO;
using JetBrains.Annotations;
using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace Italbytz.ML.UCIMLR.Tests.Unit;

[TestClass]
[TestSubject(typeof(StaticData))]
public class StaticDataTest
{
    [TestMethod]
    public void TestLoadIris()
    {
        const DatasetEnum dataset = DatasetEnum.Iris;
        var data = StaticData.Load(dataset);
        Assert.IsNotNull(data);
    }

    [TestMethod]
    public void TestLoadHeartDisease()
    {
        const DatasetEnum dataset = DatasetEnum.HeartDisease;
        var data = StaticData.Load(dataset);
        Assert.IsNotNull(data);
    }

    [TestMethod]
    public void TestLoadWineQuality()
    {
        const DatasetEnum dataset = DatasetEnum.WineQuality;
        var data = StaticData.Load(dataset);
        Assert.IsNotNull(data);
    }

    [TestMethod]
    public void TestSaveAsCsv()
    {
        const DatasetEnum dataset = DatasetEnum.Iris;
        var filePath = Path.Combine(Path.GetTempPath(), "Iris.csv");
        StaticData.SaveAsCsv(dataset, filePath);
        Assert.IsTrue(File.Exists(filePath));
        File.Delete(filePath);
    }
}