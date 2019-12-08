const BASE_URL = "https://api.unsplash.com/"
const TOKEN = "YOUR_TOKEN"
const fetch = require("node-fetch")
const fs = require("fs")
var links = []

const search = async (url, page) => {
  const response = await fetch(url + page)
    .then(res => res.json())
    .catch(error => console.log(error))
  if (response) {
    return response.results.map(e => e.links.download)
  } else {
    return []
  }
}

const collect = async (keyword, start=0) => {
  const base = `${BASE_URL}search/photos?query=${keyword}&client_id=${TOKEN}&per_page=30&page=`
  const response = await fetch(base + start)
    .then(res => res.json())
    .catch(error => console.log(error))
  const total = response.total_pages
  var end = start
  for (var i = start; i <= total; i++) {
    var curr = await search(base, i)
    if (curr.length == 0) {
      console.log("total", total)
      console.log("done")
      end = i - 1
      break
    }
    links = links.concat(curr)
  }
  var str = links.reduce((acc, e) => acc + "\n" + e, "")
  fs.writeFile(`${keyword}_${start}-${end}.csv`, str, err => {
    // In case of a error throw err.
    if (err) throw err
  })
  return
}
var args = process.argv.slice(2)
collect(args[0], args[1])
