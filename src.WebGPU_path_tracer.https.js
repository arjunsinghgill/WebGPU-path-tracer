// ASG portfolio
// gltf_viewer src https
// Arjun Singh Gill

const fs    = require('fs');
const path  = require('path');
const { URL } = require('url');
const https = require('https');

const HTTPS_TLS_CERTS = {
  key   : fs.readFileSync( path.resolve( __dirname, './certs/src.https.certs.tls.key' ) ),
  cert  : fs.readFileSync( path.resolve( __dirname, './certs/src.https.certs.tls.cert' ) ),
};

const HTTPS_font_files = {
  bruno_sc_regular : fs.readFileSync( path.resolve( __dirname, './fonts/bruno_sc_regular.ttf') ),
  raleway_medium : fs.readFileSync( path.resolve( __dirname, './fonts/raleway_medium.ttf') ),
  roboto_regular : fs.readFileSync( path.resolve( __dirname, './fonts/roboto_regular.ttf') ),
};

const HTTPS_ROUTE_home_files = {
  html : fs.readFileSync( path.resolve(__dirname,'./src.WebGPU_path_tracer.html') ),
  js   : fs.readFileSync( path.resolve(__dirname,'./src.WebGPU_path_tracer.js') ),
  css  : fs.readFileSync( path.resolve(__dirname,'./src.WebGPU_path_tracer.css') ),
};

const HTTPS_ROUTES = {
  '/': async function( IN_req, IN_res, IN_url_parsed) {
    IN_res.setHeader('Content-Type','text/html');
    IN_res.writeHead(200);
    IN_res.end( HTTPS_ROUTE_home_files.html );
  },
  '/js/home': async function( IN_req, IN_res, IN_url_parsed) {
    IN_res.setHeader('Content-Type','text/javascript');
    IN_res.writeHead(200);
    IN_res.end( HTTPS_ROUTE_home_files.js );
  },
  '/css/home': async function( IN_req, IN_res, IN_url_parsed) {
    IN_res.setHeader('Content-Type','text/css');
    IN_res.writeHead(200);
    IN_res.end( HTTPS_ROUTE_home_files.css );
  },
  '/fonts/bruno_sc_regular': async function( IN_req, IN_res, IN_url_parsed) {
    IN_res.setHeader('Content-Type','font/ttf');
    IN_res.writeHead(200);
    IN_res.end( HTTPS_font_files.bruno_sc_regular );
  },
  '/fonts/raleway_medium': async function( IN_req, IN_res, IN_url_parsed) {
    IN_res.setHeader('Content-Type','font/ttf');
    IN_res.writeHead(200);
    IN_res.end( HTTPS_font_files.raleway_medium );
  },
  '/fonts/roboto_regular': async function( IN_req, IN_res, IN_url_parsed) {
    IN_res.setHeader('Content-Type','font/ttf');
    IN_res.writeHead(200);
    IN_res.end( HTTPS_font_files.roboto_regular );
  },
};


const HTTPS_TLS_SERVER = https.createServer( HTTPS_TLS_CERTS, async (req,res) => {
  // Wrap the route handler call in a try-catch block
  try {
    // parse the url
    let TMP_url_parsed = new URL( req.url, "https://localhost:8087/" );
    // store the pathname
    let TMP_pathname = TMP_url_parsed.pathname;
    // trim the last slash in url pathname
    if( TMP_pathname[ TMP_pathname.length - 1 ] == '/' )
    {
      // remove the last forward slash
      TMP_pathname = TMP_pathname.slice(0,-1);
      // if pathname is empty, then serve the home route
      if( TMP_pathname == '' ) TMP_pathname = '/';
    }
    // if pathname in routes, serve it
    if( TMP_pathname in HTTPS_ROUTES ) {
      // serve the route
      await HTTPS_ROUTES[ TMP_pathname ]( req, res, TMP_url_parsed );
    } else {
      res.setHeader('Content-Type','text/html');
      res.writeHead(404);
      res.end('{ msg: Sorry, URL not found. }');
    }
  } catch(err) {
    res.setHeader('Content-Type','text/html');
    res.writeHead(404);
    res.end(`{ msg: Sorry, could not complete the request. }`);
  }

} ).listen(8087);

console.log('HTTPS server running at port : 8087');

