class Utils {
  static deepCopy<V> ( src: { [ k: string ]: V; } ) : { [ k: string ]: V; }
  static deepCopy<V> ( src: { [ k: number ]: V; } ) : { [ k: number ]: V; }
  static deepCopy( src: {} ) {
    var dst = <any>{};
    var keys = Object.keys( src );
    keys.map( key => dst[ key ] = src[ key ] );
    return dst;
  }
}
