// #ifdef GL_ES
// #ifdef GL_FRAGMENT_PRECISION_HIGH
// precision highp float;
// #else
precision mediump float;
//#endif
//#endif

varying vec2 vTextureCoord;
uniform sampler2D uTexture;

void main(void) {
  vec4 color = texture2D(uTexture, vTextureCoord);
  gl_FragColor = color;
}
