// 実機評価の結果をスプレッドシートに追記する Google Apps Script(ウェブアプリ)
// 記録したいスプレッドシートで「拡張機能 → Apps Script」を開き、このファイルの中身を貼り付ける。
// TOKEN は sheets_config.json の token と同じ文字列(設定済み)。他人に見せない。
// 「デプロイ → 新しいデプロイ → 種類: ウェブアプリ / 実行ユーザー: 自分 / アクセス: 全員」でデプロイする。

const TOKEN = '1e5ukihqob94z6av3fr0tsnyl78pcdwm';
const SHEET_NAME = 'results';

function doPost(e) {
  try {
    const body = JSON.parse(e.postData.contents);
    if (body.token !== TOKEN) return reply({ ok: false, error: 'bad token' });
    if (body.ping) return reply({ ok: true, ping: true });

    const lock = LockService.getScriptLock();
    lock.waitLock(20000);
    try {
      const row = body.row;
      const ss = SpreadsheetApp.getActiveSpreadsheet();
      const sh = ss.getSheetByName(SHEET_NAME) || ss.insertSheet(SHEET_NAME);

      // 見出し行(1 行目)の名前に合わせて書く。足りない列は右に足す
      let header = sh.getLastColumn() > 0 ? sh.getRange(1, 1, 1, sh.getLastColumn()).getValues()[0] : [];
      header = header.filter(String);
      const missing = Object.keys(row).filter(k => header.indexOf(k) < 0);
      if (missing.length) {
        header = header.concat(missing);
        sh.getRange(1, 1, 1, header.length).setValues([header]).setFontWeight('bold');
        sh.setFrozenRows(1);
      }
      // 同じ trial_id が既にあれば書かない(再送したときの重複を防ぐ)
      const idCol = header.indexOf('trial_id') + 1;
      if (idCol > 0 && sh.getLastRow() > 1) {
        const ids = sh.getRange(2, idCol, sh.getLastRow() - 1, 1).getValues().map(r => String(r[0]));
        if (ids.indexOf(String(row.trial_id)) >= 0) return reply({ ok: true, duplicate: true });
      }

      const values = header.map(k => (k in row ? row[k] : ''));
      sh.appendRow(values);
      return reply({ ok: true, row: sh.getLastRow() });
    } finally {
      lock.releaseLock();
    }
  } catch (err) {
    return reply({ ok: false, error: String(err) });
  }
}

function reply(obj) {
  return ContentService.createTextOutput(JSON.stringify(obj)).setMimeType(ContentService.MimeType.JSON);
}
