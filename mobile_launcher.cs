using System;
using System.Diagnostics;
using System.IO;
using System.Threading;
using System.Windows.Forms;

namespace ShelfInspectorWebLauncher
{
    internal static class Program
    {
        [STAThread]
        private static void Main()
        {
            string baseDir = AppDomain.CurrentDomain.BaseDirectory;
            string scriptPath = Path.Combine(baseDir, "mobile_server.py");
            if (!File.Exists(scriptPath))
            {
                MessageBox.Show(
                    "未找到 mobile_server.py，请确认启动器放在项目根目录。",
                    "启动失败",
                    MessageBoxButtons.OK,
                    MessageBoxIcon.Error
                );
                return;
            }

            string[] candidates = { "python.exe", "py.exe" };
            foreach (string candidate in candidates)
            {
                if (TryStart(candidate, scriptPath, baseDir))
                {
                    OpenBrowserSoon();
                    return;
                }
            }

            MessageBox.Show(
                "未能启动 Python。请确认已安装 Python，并且可以在命令行运行 python mobile_server.py。",
                "启动失败",
                MessageBoxButtons.OK,
                MessageBoxIcon.Error
            );
        }

        private static bool TryStart(string executable, string scriptPath, string workingDirectory)
        {
            try
            {
                string arguments = executable.Equals("py.exe", StringComparison.OrdinalIgnoreCase)
                    ? "-3 \"" + scriptPath + "\""
                    : "\"" + scriptPath + "\"";

                ProcessStartInfo startInfo = new ProcessStartInfo
                {
                    FileName = executable,
                    Arguments = arguments,
                    WorkingDirectory = workingDirectory,
                    UseShellExecute = false,
                    CreateNoWindow = false
                };
                Process.Start(startInfo);
                return true;
            }
            catch
            {
                return false;
            }
        }

        private static void OpenBrowserSoon()
        {
            ThreadPool.QueueUserWorkItem(_ =>
            {
                Thread.Sleep(1800);
                try
                {
                    Process.Start(new ProcessStartInfo
                    {
                        FileName = "http://127.0.0.1:5000",
                        UseShellExecute = true
                    });
                }
                catch
                {
                }
            });
        }
    }
}
