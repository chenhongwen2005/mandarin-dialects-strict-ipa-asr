@echo off
color 0E

echo ============================================================
echo   Mandarin Narrow IPA Speech Recognition
echo ============================================================
echo   Project : https://github.com/chenhongwen2005/mandarin-ipa-asr
echo   Author  : Chen Hongwen
echo   License : CC BY-NC-SA 4.0
echo ------------------------------------------------------------
echo   You ARE FREE TO:
echo     - Share  : copy and redistribute in any medium or format
echo     - Adapt  : remix, transform, and build upon the material
echo ------------------------------------------------------------
echo   UNDER THE FOLLOWING TERMS:
echo     - Attribution (BY)         : credit the original author,
echo                                  keep project source and license,
echo                                  indicate if changes were made
echo     - NonCommercial (NC)       : NOT for commercial purposes
echo     - ShareAlike (SA)          : derivative works MUST be
echo                                  released under the same license
echo     - No additional restrictions: no legal or technical
echo                                  limits on others' rights
echo ------------------------------------------------------------
echo   YOU MAY NOT:
echo     - Use for commercial purposes
echo     - Re-license derivatives under MIT / Apache / etc.
echo     - Add DRM or other technical restrictions
echo     - Remove or alter author credit / license info
echo ------------------------------------------------------------
echo   Full license:
echo   https://creativecommons.org/licenses/by-nc-sa/4.0/
echo ============================================================
echo.

runtime\python.exe app.py --ckpt weights\best.pt

pause